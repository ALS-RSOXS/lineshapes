import { copyFileSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = dirname(fileURLToPath(import.meta.url));
const crateDir = resolve(scriptDir, "..");
const workspaceRoot = resolve(crateDir, "../..");
const pkgDir = resolve(crateDir, "pkg");
const tsDir = resolve(crateDir, "ts");

function run(cmd: string[], cwd: string) {
  const proc = Bun.spawnSync(cmd, {
    cwd,
    stdout: "inherit",
    stderr: "inherit",
  });
  if (proc.exitCode !== 0) {
    throw new Error(`Command failed (${proc.exitCode}): ${cmd.join(" ")}`);
  }
}

function readWorkspaceVersion(): string {
  const cargoToml = readFileSync(resolve(workspaceRoot, "Cargo.toml"), "utf-8");
  const match = cargoToml.match(
    /\[workspace\.package\][\s\S]*?^\s*version\s*=\s*"([^"]+)"/m,
  );
  if (!match?.[1]) {
    throw new Error("Could not read [workspace.package] version from Cargo.toml");
  }
  return match[1];
}

// 0. Sync version in crates/lineshapes-wasm/package.json from root Cargo.toml
// and copy LICENSE + COPYRIGHT from workspace root.
const version = readWorkspaceVersion();
const pkgJsonPath = resolve(crateDir, "package.json");
const pkgJson = JSON.parse(readFileSync(pkgJsonPath, "utf-8"));
if (pkgJson.version !== version) {
  pkgJson.version = version;
  writeFileSync(pkgJsonPath, `${JSON.stringify(pkgJson, null, 2)}\n`);
}
copyFileSync(resolve(workspaceRoot, "LICENSE"), resolve(crateDir, "LICENSE"));
copyFileSync(
  resolve(workspaceRoot, "COPYRIGHT"),
  resolve(crateDir, "COPYRIGHT"),
);

mkdirSync(pkgDir, { recursive: true });

// 1. Compile the Rust crate to wasm32-unknown-unknown (release)
run(
  [
    "cargo",
    "build",
    "--target",
    "wasm32-unknown-unknown",
    "--release",
    "-p",
    "lineshapes-wasm",
  ],
  workspaceRoot,
);

// 2. Generate JS + TS bindings with wasm-bindgen
const wasmInput = resolve(
  workspaceRoot,
  "target/wasm32-unknown-unknown/release/lineshapes_wasm.wasm",
);
run(
  ["wasm-bindgen", wasmInput, "--out-dir", pkgDir, "--target", "web"],
  workspaceRoot,
);

// Ensure generated .d.ts works with TypeScript projects targeting `"lib": ["dom", "ES2022"]`
const rawDtsPath = resolve(pkgDir, "lineshapes_wasm.d.ts");
const rawDts = readFileSync(rawDtsPath, "utf-8").replace(
  /^\s*\[Symbol\.dispose\]\(\): void;\n/gm,
  "",
);
writeFileSync(rawDtsPath, rawDts);

// 3. Emit embedded base64 WASM loader so initialization works synchronously and
// asynchronously in Bun, Node, Next.js (Turbopack/Webpack), and browsers with zero asset config.
const wasmBgPath = resolve(pkgDir, "lineshapes_wasm_bg.wasm");
const wasmBytes = readFileSync(wasmBgPath);
const wasmBase64 = wasmBytes.toString("base64");

writeFileSync(
  resolve(pkgDir, "wasm_bytes.js"),
  `const WASM_BASE64 = ${JSON.stringify(wasmBase64)};

let cachedBytes = null;

/**
 * Decode and cache the embedded \`lineshapes_wasm_bg.wasm\` binary bytes.
 * @returns {Uint8Array}
 */
export function getWasmBytes() {
  if (cachedBytes !== null) return cachedBytes;
  if (typeof Uint8Array.fromBase64 === "function") {
    cachedBytes = Uint8Array.fromBase64(WASM_BASE64);
  } else if (typeof Buffer !== "undefined") {
    cachedBytes = new Uint8Array(Buffer.from(WASM_BASE64, "base64"));
  } else {
    const bin = atob(WASM_BASE64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) {
      bytes[i] = bin.charCodeAt(i);
    }
    cachedBytes = bytes;
  }
  return cachedBytes;
}
`,
);

writeFileSync(
  resolve(pkgDir, "wasm_bytes.d.ts"),
  `export function getWasmBytes(): Uint8Array;\n`,
);

// 4. Copy the high-level TypeScript/ESM wrapper into pkg/
copyFileSync(resolve(tsDir, "index.js"), resolve(pkgDir, "index.js"));
copyFileSync(resolve(tsDir, "index.d.ts"), resolve(pkgDir, "index.d.ts"));

console.log(
  `Built lineshapes (npm) v${version} -> ${pkgDir} (${(wasmBytes.byteLength / 1024).toFixed(1)} KB wasm)`,
);
