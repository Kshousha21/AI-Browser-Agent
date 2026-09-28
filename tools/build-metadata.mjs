// Local build fingerprints and scope parity; never reads environment files.
import { createHash } from "node:crypto";
import { existsSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { dirname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { execFileSync } from "node:child_process";
import { createRequire } from "node:module";
const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const dist = join(root, "extension/dist");
function files(path) {
  return readdirSync(path, {withFileTypes:true}).flatMap(entry => {
    const value = join(path, entry.name);
    if (entry.isSymbolicLink()) throw Error("Symbolic links are not build inputs");
    return entry.isDirectory() ? files(value) : [value];
  });
}
function fingerprint(paths) {
  const hash = createHash("sha256");
  for (const path of [...paths].sort()) hash.update(relative(root,path).replaceAll("\\","/")).update("\0").update(readFileSync(path)).update("\0");
  return hash.digest("hex");
}
function inputs() {
  return [...["extension/src","extension/public","shared"].flatMap(path=>files(join(root,path))),
    ...["extension/package.json","extension/package-lock.json","extension/tsconfig.json","extension/vite.config.ts","extension/vite.content.config.ts","tools/build-metadata.mjs"].map(path=>join(root,path))];
}
function outputs() { return files(dist).filter(path=>path!==join(dist,"build-info.json")); }
function gitCommit() {
  try {
    if (!existsSync(join(root,".git")) || ["GIT_DIR","GIT_WORK_TREE","GIT_COMMON_DIR"].some(name => process.env[name] !== undefined)) return null;
    // Git proves worktree-root identity even when its absolute spelling is UNC.
    // Never replace the original execution directory with an absolute Git path.
    const identity = execFileSync("git",["-C",root,"rev-parse","--is-inside-work-tree","--show-prefix"],{encoding:"utf8",stdio:["ignore","pipe","ignore"],timeout:3000});
    if (identity.replaceAll("\r\n","\n") !== "true\n\n") return null;
    const sha = execFileSync("git",["-C",root,"rev-parse","--short=12","HEAD"],{encoding:"utf8",stdio:["ignore","pipe","ignore"],timeout:3000}).trim();
    return /^[0-9a-f]{7,40}$/.test(sha) ? sha : null;
  } catch { return null; }
}
function scopeParity() {
  const require = createRequire(join(root,"extension/package.json"));
  const {buildSync} = require("esbuild");
  const code=buildSync({absWorkingDir:root,entryPoints:[join(root,"extension/src/lib/product-scope.ts")],bundle:true,write:false,format:"cjs",platform:"node"}).outputFiles[0].text;
  const module={exports:{}};
  new Function("module","exports","require",code)(module,module.exports,require);
  const samples=JSON.parse(readFileSync(join(root,"shared/scope-diagnostics.json"),"utf8"));
  return samples.every(row=>module.exports.isProductUrl(row.url)===row.allowed);
}
try {
  const contract=JSON.parse(readFileSync(join(root,"shared/build-contract.json"),"utf8"));
  if (process.argv.includes("--scope")) {
    const compatible=scopeParity(); console.log(JSON.stringify({scope_parity:compatible})); process.exitCode=compatible?0:1;
  } else if (process.argv.includes("--check")) {
    const saved=JSON.parse(readFileSync(join(dist,"build-info.json"),"utf8"));
    const fresh=saved.source_fingerprint===fingerprint(inputs()) && saved.output_fingerprint===fingerprint(outputs());
    const compatible=saved.protocol_revision===contract.protocol_revision && saved.protocol_version===contract.protocol_version;
    console.log(JSON.stringify({fresh,compatible,application_version:typeof saved.application_version === "string" && /^[0-9]+\.[0-9]+\.[0-9]+$/.test(saved.application_version) ? saved.application_version : null,protocol_version:Number.isSafeInteger(saved.protocol_version) ? saved.protocol_version : null,protocol_revision:Number.isSafeInteger(saved.protocol_revision) ? saved.protocol_revision : null}));
    process.exitCode=fresh&&compatible?0:1;
  } else if (process.argv.includes("--write")) {
    if (!existsSync(join(dist,"manifest.json")) || !existsSync(join(dist,"content/extract.js"))) throw Error("Build incomplete");
    const data={...contract,git_commit:gitCommit(),source_fingerprint:fingerprint(inputs()),output_fingerprint:fingerprint(outputs())};
    writeFileSync(join(dist,"build-info.json"),JSON.stringify(data,null,2)+"\n");
    console.log(`Build metadata: application ${data.application_version}; protocol ${data.protocol_version}/${data.protocol_revision}; commit ${data.git_commit??"unavailable"}`);
  } else throw Error("Choose --write, --check, or --scope");
} catch {
  console.error("Build metadata check failed. Run scripts/build.ps1; ensure source files and dependencies are present.");
  process.exitCode=1;
}
