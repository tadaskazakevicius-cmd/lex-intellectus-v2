import { cp, rm } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const uiRoot = path.resolve(__dirname, "..");
const distDir = path.resolve(uiRoot, "dist");

const serverSpaDir = path.resolve(uiRoot, "..", "server", "src", "lex_server", "static", "spa");

async function main() {
  // Replace existing embedded SPA build.
  await rm(serverSpaDir, { recursive: true, force: true });
  await cp(distDir, serverSpaDir, { recursive: true });
  process.stdout.write(`Copied SPA build to: ${serverSpaDir}\n`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

