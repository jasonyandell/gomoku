import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import * as fs from "node:fs";
import * as fsp from "node:fs/promises";
import * as path from "node:path";
import type { ExtensionAPI, ExtensionCommandContext } from "@mariozechner/pi-coding-agent";

type ResourceName = "gpu" | "cpu" | "code" | "wiki" | string;

type FrontierConfig = {
	template: string;
	maxWorkers: number;
	resourceLimits: Record<string, number>;
	defaultHeats: string[];
	baseBranch?: string;
	wikiFiles: string[];
	workerEnv?: Record<string, string>;
	validationCommands?: Record<string, string>;
	receiptRequiredFields?: string[];
};

type Lane = {
	id: string;
	title: string;
	stage: string;
	heat: string;
	resource: ResourceName;
	goal: string;
	nextAction: string;
	doneShape?: string[];
	blocked?: boolean;
};

type RunOptions = {
	maxWorkers: number;
	heats: string[];
	laneFilters: string[];
	resources: string[];
	model?: string;
	curatorModel?: string;
	merge: boolean;
	curate: boolean;
	dryRun: boolean;
	yes: boolean;
	allowDirty: boolean;
	keepWorktrees: boolean;
	validate?: string;
};

type WorkerStatus = "queued" | "running" | "completed" | "failed" | "integrated" | "blocked" | "skipped";
type IntegrationStatus = "pending" | "no_changes" | "merged" | "conflict" | "failed" | "skipped";

type WorkerState = {
	lane: Lane;
	slug: string;
	branch: string;
	worktreePath: string;
	promptPath: string;
	receiptPath: string;
	sessionDir: string;
	status: WorkerStatus;
	integrationStatus: IntegrationStatus;
	pid?: number;
	startedAt?: string;
	finishedAt?: string;
	exitCode?: number;
	stderr?: string;
	lastEvent?: string;
	lastText?: string;
	finalText?: string;
	commit?: string;
	integratedCommit?: string;
	blocker?: string;
};

type FrontierRun = {
	id: string;
	cwd: string;
	baseBranch: string;
	rootDir: string;
	worktreeRoot: string;
	startedAt: string;
	finishedAt?: string;
	status: "planning" | "running" | "integrating" | "curating" | "completed" | "failed" | "stopped";
	config: FrontierConfig;
	options: RunOptions;
	workers: WorkerState[];
	managerPlanPath: string;
	curatorPromptPath?: string;
	curatorExitCode?: number;
	curatorFinalText?: string;
	curatorStderr?: string;
	validationResults?: Array<{ worker?: string; commandName: string; command: string; exitCode: number; stdout: string; stderr: string }>;
};

type ExecResult = { code: number; stdout: string; stderr: string };

const CONFIG_PATH = ".frontier/config.json";
const LANES_PATH = ".frontier/lanes.json";
const DEFAULT_WIKI_FILES = ["AGENTS.md", "wiki/index.md", "TRAINING_WIKI.md", "wiki/ops/status.md", "wiki/ops/frontier.md", "wiki/ops/experiment-ledger.md", "wiki/ops/test-ledger.md"];

let activeRun: FrontierRun | undefined;
let currentCtx: ExtensionCommandContext | undefined;
const childProcesses = new Map<string, ChildProcessWithoutNullStreams>();

function nowIso(): string {
	return new Date().toISOString();
}

function compactRunId(): string {
	return new Date().toISOString().replace(/[-:]/g, "").replace(/\.\d{3}Z$/, "Z");
}

function slugify(input: string, maxLength = 56): string {
	return (
		input
			.toLowerCase()
			.replace(/[`'"()[\]/]/g, "")
			.replace(/[^a-z0-9]+/g, "-")
			.replace(/^-+|-+$/g, "")
			.slice(0, maxLength)
			.replace(/-+$/g, "") || "lane"
	);
}

function shellSplit(input: string): string[] {
	const tokens: string[] = [];
	const re = /"((?:\\.|[^"])*)"|'((?:\\.|[^'])*)'|(\S+)/g;
	let match: RegExpExecArray | null;
	while ((match = re.exec(input))) {
		const raw = match[1] ?? match[2] ?? match[3] ?? "";
		tokens.push(raw.replace(/\\(["'])/g, "$1"));
	}
	return tokens;
}

function parseArgs(args: string, config?: FrontierConfig): RunOptions {
	const tokens = shellSplit(args);
	const options: RunOptions = {
		maxWorkers: config?.maxWorkers ?? 5,
		heats: config?.defaultHeats ?? ["hot", "active", "warm"],
		laneFilters: [],
		resources: [],
		merge: true,
		curate: true,
		dryRun: false,
		yes: false,
		allowDirty: false,
		keepWorktrees: false,
	};

	for (let i = 0; i < tokens.length; i++) {
		const token = tokens[i];
		const next = () => tokens[++i];
		if (token === "--dry-run") options.dryRun = true;
		else if (token === "--yes" || token === "-y") options.yes = true;
		else if (token === "--no-merge") options.merge = false;
		else if (token === "--no-curator") options.curate = false;
		else if (token === "--allow-dirty") options.allowDirty = true;
		else if (token === "--keep-worktrees") options.keepWorktrees = true;
		else if (token.startsWith("--max=")) options.maxWorkers = Number(token.slice("--max=".length));
		else if (token === "--max") options.maxWorkers = Number(next());
		else if (token.startsWith("--heat=")) options.heats = splitCsv(token.slice("--heat=".length));
		else if (token === "--heat") options.heats = splitCsv(next() ?? "");
		else if (token.startsWith("--lane=")) options.laneFilters.push(token.slice("--lane=".length));
		else if (token === "--lane") options.laneFilters.push(next() ?? "");
		else if (token.startsWith("--resource=")) options.resources.push(...splitCsv(token.slice("--resource=".length)));
		else if (token === "--resource") options.resources.push(...splitCsv(next() ?? ""));
		else if (token.startsWith("--model=")) options.model = token.slice("--model=".length);
		else if (token === "--model") options.model = next();
		else if (token.startsWith("--curator-model=")) options.curatorModel = token.slice("--curator-model=".length);
		else if (token === "--curator-model") options.curatorModel = next();
		else if (token.startsWith("--validate=")) options.validate = token.slice("--validate=".length);
		else if (token === "--validate") options.validate = next();
		else if (token.trim()) options.laneFilters.push(token);
	}

	if (!Number.isFinite(options.maxWorkers) || options.maxWorkers < 1) options.maxWorkers = config?.maxWorkers ?? 5;
	options.maxWorkers = Math.min(5, Math.floor(options.maxWorkers));
	options.laneFilters = options.laneFilters.map((f) => f.trim()).filter(Boolean);
	options.resources = options.resources.map((f) => f.trim().toLowerCase()).filter(Boolean);
	return options;
}

function splitCsv(input: string): string[] {
	return input.split(",").map((s) => s.trim()).filter(Boolean);
}

async function pathExists(filePath: string): Promise<boolean> {
	try {
		await fsp.access(filePath);
		return true;
	} catch {
		return false;
	}
}

async function readText(filePath: string): Promise<string> {
	return fsp.readFile(filePath, "utf8");
}

async function writeText(filePath: string, content: string): Promise<void> {
	await fsp.mkdir(path.dirname(filePath), { recursive: true });
	await fsp.writeFile(filePath, content, "utf8");
}

async function readJson<T>(filePath: string): Promise<T> {
	return JSON.parse(await readText(filePath)) as T;
}

async function readConfig(cwd: string): Promise<FrontierConfig> {
	const filePath = path.join(cwd, CONFIG_PATH);
	if (!(await pathExists(filePath))) {
		return {
			template: "generic",
			maxWorkers: 5,
			resourceLimits: { gpu: 1, cpu: 3, code: 3, wiki: 1 },
			defaultHeats: ["hot", "active", "warm"],
			wikiFiles: DEFAULT_WIKI_FILES,
		};
	}
	const config = await readJson<FrontierConfig>(filePath);
	return {
		...config,
		maxWorkers: Math.min(5, config.maxWorkers ?? 5),
		resourceLimits: config.resourceLimits ?? { gpu: 1, cpu: 3, code: 3, wiki: 1 },
		defaultHeats: config.defaultHeats ?? ["hot", "active", "warm"],
		wikiFiles: config.wikiFiles ?? DEFAULT_WIKI_FILES,
	};
}

async function readLanes(cwd: string): Promise<Lane[]> {
	const filePath = path.join(cwd, LANES_PATH);
	if (!(await pathExists(filePath))) throw new Error(`Missing ${LANES_PATH}. Run /frontier-init ml-perf first.`);
	return readJson<Lane[]>(filePath);
}

function execFile(command: string, args: string[], cwd: string, timeoutMs = 120_000, env?: Record<string, string>): Promise<ExecResult> {
	return new Promise((resolve) => {
		const proc = spawn(command, args, {
			cwd,
			shell: false,
			stdio: ["ignore", "pipe", "pipe"],
			env: { ...process.env, ...env },
		});
		let stdout = "";
		let stderr = "";
		let settled = false;
		const timeout = setTimeout(() => {
			if (!settled) {
				stderr += `\nTimed out after ${timeoutMs}ms`;
				proc.kill("SIGTERM");
				setTimeout(() => {
					if (!proc.killed) proc.kill("SIGKILL");
				}, 5000).unref();
			}
		}, timeoutMs);
		proc.stdout.on("data", (chunk) => (stdout += chunk.toString()));
		proc.stderr.on("data", (chunk) => (stderr += chunk.toString()));
		proc.on("error", (error) => {
			if (settled) return;
			settled = true;
			clearTimeout(timeout);
			resolve({ code: 1, stdout, stderr: stderr + String(error) });
		});
		proc.on("close", (code) => {
			if (settled) return;
			settled = true;
			clearTimeout(timeout);
			resolve({ code: code ?? 0, stdout, stderr });
		});
	});
}

function execShell(command: string, cwd: string, timeoutMs = 600_000, env?: Record<string, string>): Promise<ExecResult> {
	return execFile("bash", ["-lc", command], cwd, timeoutMs, env);
}

async function git(cwd: string, args: string[], timeoutMs = 120_000): Promise<ExecResult> {
	return execFile("git", args, cwd, timeoutMs);
}

async function currentBranch(cwd: string): Promise<string> {
	const result = await git(cwd, ["branch", "--show-current"]);
	return result.stdout.trim() || "HEAD";
}

async function porcelainStatus(cwd: string): Promise<string> {
	const result = await git(cwd, ["status", "--porcelain", "--untracked-files=all"], 30_000);
	return result.stdout;
}

function isIgnorableDirtyLine(line: string): boolean {
	const file = line.slice(3).replace(/^"|"$/g, "");
	return file.startsWith(".frontier/runs/") || file.startsWith(".frontier/worktrees/") || file.startsWith(".frontier/tmp/");
}

async function shouldIgnoreDirtyLine(cwd: string, line: string): Promise<boolean> {
	if (isIgnorableDirtyLine(line)) return true;
	// Some harnesses create an empty root CLAUDE.md as local context scratch.
	// Ignore only the empty untracked variant; a non-empty or tracked CLAUDE.md still blocks.
	if (line.trim() === "?? CLAUDE.md") {
		try {
			const stat = await fsp.stat(path.join(cwd, "CLAUDE.md"));
			return stat.size === 0;
		} catch {
			return false;
		}
	}
	return false;
}

async function assertBaseClean(cwd: string, allowDirty: boolean): Promise<{ clean: boolean; blockingStatus: string }> {
	const status = await porcelainStatus(cwd);
	const lines = status.split("\n").map((line) => line.trimEnd()).filter(Boolean);
	const blockingLines: string[] = [];
	for (const line of lines) {
		if (!(await shouldIgnoreDirtyLine(cwd, line))) blockingLines.push(line);
	}
	return { clean: allowDirty || blockingLines.length === 0, blockingStatus: blockingLines.join("\n") };
}

function heatRank(heat: string): number {
	const ranks: Record<string, number> = { hot: 0, active: 1, warm: 2, cool: 3, seeded: 4 };
	return ranks[heat.toLowerCase()] ?? 99;
}

function selectLanes(lanes: Lane[], options: RunOptions): Lane[] {
	const heatSet = new Set(options.heats.map((h) => h.toLowerCase()));
	const filters = options.laneFilters.map((f) => f.toLowerCase());
	const resources = new Set(options.resources);
	return lanes
		.filter((lane) => !lane.blocked && !/\b(done|complete|closed|obsolete|blocked)\b/i.test(`${lane.stage} ${lane.heat}`))
		.filter((lane) => filters.length === 0 || filters.some((filter) => `${lane.id} ${lane.title} ${lane.goal} ${lane.nextAction}`.toLowerCase().includes(filter)))
		.filter((lane) => filters.length > 0 || heatSet.has(lane.heat.toLowerCase()))
		.filter((lane) => resources.size === 0 || resources.has(String(lane.resource).toLowerCase()))
		.sort((a, b) => heatRank(a.heat) - heatRank(b.heat))
		.slice(0, options.maxWorkers);
}

function runRoot(cwd: string, runId: string): string {
	return path.join(cwd, ".frontier", "runs", runId);
}

function worktreeRoot(cwd: string): string {
	return path.join(cwd, ".frontier", "worktrees");
}

async function saveRun(run: FrontierRun): Promise<void> {
	await writeText(path.join(run.rootDir, "state.json"), JSON.stringify(run, null, 2) + "\n");
}

async function appendEvent(run: FrontierRun, event: Record<string, unknown>): Promise<void> {
	await fsp.mkdir(run.rootDir, { recursive: true });
	await fsp.appendFile(path.join(run.rootDir, "events.jsonl"), JSON.stringify({ ts: nowIso(), ...event }) + "\n");
}

function getPiInvocation(args: string[]): { command: string; args: string[] } {
	const currentScript = process.argv[1];
	const isBunVirtualScript = currentScript?.startsWith("/$bunfs/root/");
	if (currentScript && !isBunVirtualScript && fs.existsSync(currentScript)) return { command: process.execPath, args: [currentScript, ...args] };
	const execName = path.basename(process.execPath).toLowerCase();
	if (!/^(node|bun)(\.exe)?$/.test(execName)) return { command: process.execPath, args };
	return { command: "pi", args };
}

function textFromContent(content: any): string {
	if (typeof content === "string") return content;
	if (!Array.isArray(content)) return "";
	return content
		.map((part) => {
			if (part?.type === "text") return part.text ?? "";
			if (part?.type === "toolCall") return `[tool:${part.name}]`;
			return "";
		})
		.filter(Boolean)
		.join("\n");
}

function makeWorkerPrompt(run: FrontierRun, worker: WorkerState): string {
	const openNote = `wiki/ops/open-notes/${run.id}-${worker.slug}.md`;
	const doneShape = worker.lane.doneShape?.map((item) => `- ${item}`).join("\n") || "- receipt-backed decision";
	const commands = run.config.validationCommands
		? Object.entries(run.config.validationCommands).map(([name, command]) => `- ${name}: \`${command}\``).join("\n")
		: "- pytest -q";
	return `You are a frontier worker in an isolated git worktree for an ML performance project.

Use these skills if available and relevant: frontier-orchestration, ml-performance-lab, experiment-ledger.

Run ID: ${run.id}
Assigned lane: ${worker.lane.id} — ${worker.lane.title}
Stage: ${worker.lane.stage}
Heat: ${worker.lane.heat}
Resource class: ${worker.lane.resource}
Goal: ${worker.lane.goal}
Next action: ${worker.lane.nextAction}

Paths:
- Worker cwd/worktree: ${worker.worktreePath}
- Main project worktree, read-only for you: ${run.cwd}
- External receipt path for manager: ${worker.receiptPath}
- Open note path in repo: ${openNote}

Read first:
${run.config.wikiFiles.map((file) => `- ${file}`).join("\n")}

Hard constraints:
- Work only this assigned lane. Do not pick another frontier.
- You are one of several concurrent workers; avoid broad rewrites and unrelated cleanup.
- Do not edit the main worktree at ${run.cwd}.
- Prefer bounded benchmark/design work over long training. Do not launch a long live run unless the lane explicitly requires it and the receipt explains why.
- Do not chase Activity Monitor GPU percent. Score performance by paired wall-clock, games/sec, positions/sec, eval time, and training-quality guardrails.
- Treat short evals as noisy. Strength claims need fixed baseline or archive evidence plus uncertainty.
- Shared board files are curator-owned: wiki/ops/status.md, wiki/ops/frontier.md, wiki/ops/baselines.md, wiki/ops/experiment-ledger.md, wiki/ops/test-ledger.md. Prefer an open note plus board-update recommendations.

Lane done shape:
${doneShape}

Useful validation commands:
${commands}

Receipt schema required:
\`\`\`yaml
lane:
hypothesis:
code_ref:
dataset_ref:
baseline_command:
candidate_command:
hardware:
seed:
baseline_metric:
candidate_metric:
delta:
confidence:
artifacts:
commands_run:
decision: promote | reject | blocked | needs_repeat
next_action:
\`\`\`

Before finishing:
1. Write ${openNote} with the receipt, files touched, commands run, result, blocker if any, and board-update recommendation.
2. Write the same compact final receipt to ${worker.receiptPath} for the manager.
3. If you changed files, commit one focused commit on branch ${worker.branch}.
4. Finish with lane, commit hash or no-commit, metrics/result, decision, and next action.`;
}

function makeCuratorPrompt(run: FrontierRun): string {
	const receiptList = run.workers.map((worker) => `- ${worker.lane.id} (${worker.status}/${worker.integrationStatus}): ${worker.receiptPath}`).join("\n");
	return `You are the wiki curator for frontier run ${run.id}.

Use these skills if available: wiki-curation, frontier-orchestration, ml-performance-lab, experiment-ledger.

Work in main project cwd: ${run.cwd}

Read first:
${run.config.wikiFiles.map((file) => `- ${file}`).join("\n")}

Then read:
- ${path.join(run.rootDir, "state.json")}
${receiptList}

Curator rules:
- Keep raw evidence separate from maintained synthesis.
- Roll completed/integrated receipts into wiki/ops/status.md, wiki/ops/frontier.md, wiki/ops/baselines.md, wiki/ops/experiment-ledger.md, and wiki/ops/test-ledger.md.
- Preserve worker detail in wiki/ops/open-notes/.
- Update wiki/log.md if maintained synthesis or structure changed materially.
- Mark blockers honestly; do not promote claims without benchmark evidence.
- Do not commit. The frontier manager will commit curator changes.

Finish with files changed, receipts integrated, blockers, promoted next lane, and validation/test ledger updates.`;
}

async function prepareWorker(run: FrontierRun, lane: Lane, index: number): Promise<WorkerState> {
	const slug = `${String(index + 1).padStart(2, "0")}-${slugify(lane.id || lane.title)}`;
	const branch = `frontier/${run.id}/${slug}`;
	const worktreePath = path.join(run.worktreeRoot, `${run.id}-${slug}`);
	const workerDir = path.join(run.rootDir, "workers", slug);
	const promptPath = path.join(workerDir, "prompt.md");
	const receiptPath = path.join(workerDir, "receipt.md");
	const sessionDir = path.join(workerDir, "sessions");
	await fsp.mkdir(workerDir, { recursive: true });
	await fsp.mkdir(sessionDir, { recursive: true });
	const worker: WorkerState = { lane, slug, branch, worktreePath, promptPath, receiptPath, sessionDir, status: "queued", integrationStatus: "pending" };
	const added = await git(run.cwd, ["worktree", "add", worktreePath, "-b", branch, "HEAD"], 120_000);
	if (added.code !== 0) {
		worker.status = "failed";
		worker.integrationStatus = "failed";
		worker.blocker = `git worktree add failed: ${added.stderr || added.stdout}`;
		return worker;
	}
	await writeText(promptPath, makeWorkerPrompt(run, worker));
	return worker;
}

async function writeManagerPlan(run: FrontierRun): Promise<void> {
	const body = `# Frontier Run ${run.id}\n\nStarted: ${run.startedAt}\nCwd: ${run.cwd}\nBranch: ${run.baseBranch}\nTemplate: ${run.config.template}\nMax workers: ${run.options.maxWorkers}\nMerge: ${run.options.merge}\nCurator: ${run.options.curate}\n\n## Claimed Lanes\n\n${run.workers.map((worker) => `- ${worker.slug}: ${worker.lane.title} (${worker.lane.heat}/${worker.lane.resource})\n  - branch: \`${worker.branch}\`\n  - worktree: \`${worker.worktreePath}\`\n  - next: ${worker.lane.nextAction}`).join("\n")}\n\n## Resource Limits\n\n${Object.entries(run.config.resourceLimits).map(([name, limit]) => `- ${name}: ${limit}`).join("\n")}\n`;
	await writeText(run.managerPlanPath, body);
}

function workerEnv(run: FrontierRun, worker: WorkerState): Record<string, string> {
	return {
		...(run.config.workerEnv ?? {}),
		FRONTIER_RUN_ID: run.id,
		FRONTIER_LANE_ID: worker.lane.id,
		FRONTIER_RESOURCE: String(worker.lane.resource),
	};
}

async function runPiJson(cwd: string, promptPath: string, sessionDir: string, model: string | undefined, env: Record<string, string>, onEvent: (event: any) => Promise<void> | void): Promise<{ exitCode: number; stderr: string; finalText: string }> {
	const args = ["--mode", "json", "--no-extensions", "--session-dir", sessionDir];
	if (model) args.push("--model", model);
	args.push("-p", `@${promptPath}`);
	const invocation = getPiInvocation(args);

	return new Promise((resolve) => {
		const proc = spawn(invocation.command, invocation.args, { cwd, shell: false, stdio: ["ignore", "pipe", "pipe"], env: { ...process.env, ...env } });
		const key = `${cwd}:${promptPath}`;
		childProcesses.set(key, proc);
		let buffer = "";
		let stderr = "";
		let finalText = "";
		let settled = false;
		const processLine = (line: string) => {
			if (!line.trim()) return;
			let event: any;
			try {
				event = JSON.parse(line);
			} catch {
				return;
			}
			if (event.type === "message_end" && event.message?.role === "assistant") {
				const text = textFromContent(event.message.content).trim();
				if (text) finalText = text;
			}
			void onEvent(event);
		};
		proc.stdout.on("data", (chunk) => {
			buffer += chunk.toString();
			const lines = buffer.split("\n");
			buffer = lines.pop() ?? "";
			for (const line of lines) processLine(line);
		});
		proc.stderr.on("data", (chunk) => (stderr += chunk.toString()));
		proc.on("error", (error) => {
			if (settled) return;
			settled = true;
			childProcesses.delete(key);
			resolve({ exitCode: 1, stderr: stderr + String(error), finalText });
		});
		proc.on("close", (code) => {
			if (settled) return;
			settled = true;
			if (buffer.trim()) processLine(buffer);
			childProcesses.delete(key);
			resolve({ exitCode: code ?? 0, stderr, finalText });
		});
	});
}

async function runWorker(run: FrontierRun, worker: WorkerState): Promise<void> {
	if (worker.status === "failed") return;
	worker.status = "running";
	worker.startedAt = nowIso();
	await appendEvent(run, { type: "worker_start", worker: worker.slug, lane: worker.lane.id, resource: worker.lane.resource });
	await saveRun(run);
	const result = await runPiJson(worker.worktreePath, worker.promptPath, worker.sessionDir, run.options.model, workerEnv(run, worker), async (event) => {
		await fsp.appendFile(path.join(run.rootDir, "workers", worker.slug, "events.jsonl"), JSON.stringify(event) + "\n");
		if (event.type === "tool_execution_start") worker.lastEvent = `tool ${event.toolName}`;
		if (event.type === "message_update" && event.assistantMessageEvent?.type === "text_delta") {
			worker.lastText = ((worker.lastText ?? "") + (event.assistantMessageEvent.delta ?? "")).slice(-500);
		}
		if (event.type === "message_end" && event.message?.role === "assistant") {
			const text = textFromContent(event.message.content).trim();
			if (text) worker.finalText = text;
		}
		await saveRun(run);
	});
	worker.exitCode = result.exitCode;
	worker.stderr = result.stderr;
	worker.finalText = result.finalText || worker.finalText;
	worker.finishedAt = nowIso();
	worker.status = result.exitCode === 0 ? "completed" : "failed";
	await appendEvent(run, { type: "worker_end", worker: worker.slug, lane: worker.lane.id, exitCode: result.exitCode });
	await saveRun(run);
}

async function runWorkersWithResources(run: FrontierRun): Promise<void> {
	const queued = [...run.workers];
	const running = new Map<Promise<void>, WorkerState>();
	const counts: Record<string, number> = {};
	const totalRunning = () => running.size;
	const canLaunch = (worker: WorkerState) => {
		const resource = String(worker.lane.resource || "cpu").toLowerCase();
		const limit = run.config.resourceLimits[resource] ?? run.options.maxWorkers;
		return totalRunning() < run.options.maxWorkers && (counts[resource] ?? 0) < limit;
	};
	const launch = (worker: WorkerState) => {
		const resource = String(worker.lane.resource || "cpu").toLowerCase();
		counts[resource] = (counts[resource] ?? 0) + 1;
		const promise = runWorker(run, worker).finally(() => {
			counts[resource] = Math.max(0, (counts[resource] ?? 1) - 1);
			running.delete(promise);
		});
		running.set(promise, worker);
	};

	while (queued.length > 0 || running.size > 0) {
		let launched = false;
		for (let i = 0; i < queued.length; ) {
			const worker = queued[i];
			if (canLaunch(worker)) {
				queued.splice(i, 1);
				launch(worker);
				launched = true;
			} else {
				i++;
			}
		}
		if (!launched && running.size > 0) await Promise.race(running.keys());
		if (activeRun?.status === "stopped") return;
	}
}

async function commitDirtyWorktree(worker: WorkerState): Promise<void> {
	const status = await porcelainStatus(worker.worktreePath);
	if (!status.trim()) return;
	await git(worker.worktreePath, ["add", "-A"], 60_000);
	const commit = await git(worker.worktreePath, ["commit", "-m", `frontier(${worker.lane.id}): ${worker.lane.title}`], 120_000);
	if (commit.code !== 0 && !/nothing to commit/i.test(commit.stderr + commit.stdout)) throw new Error(commit.stderr || commit.stdout || "git commit failed");
}

async function runValidationIfRequested(run: FrontierRun, worker: WorkerState): Promise<void> {
	const name = run.options.validate;
	if (!name) return;
	const command = run.config.validationCommands?.[name] ?? name;
	if (!command) return;
	const result = await execShell(command, run.cwd, 900_000, run.config.workerEnv);
	run.validationResults ??= [];
	run.validationResults.push({ worker: worker.slug, commandName: name, command, exitCode: result.code, stdout: result.stdout.slice(-5000), stderr: result.stderr.slice(-5000) });
	if (result.code !== 0) {
		worker.status = "blocked";
		worker.integrationStatus = "failed";
		worker.blocker = `Validation failed (${name}): ${result.stderr || result.stdout}`;
	}
}

async function integrateWorker(run: FrontierRun, worker: WorkerState): Promise<void> {
	if (!run.options.merge) {
		worker.integrationStatus = "skipped";
		if (worker.status === "completed") worker.status = "skipped";
		return;
	}
	if (worker.status !== "completed") {
		worker.integrationStatus = "skipped";
		return;
	}
	const clean = await assertBaseClean(run.cwd, false);
	if (!clean.clean) {
		worker.status = "blocked";
		worker.integrationStatus = "failed";
		worker.blocker = `Base worktree dirty before merge:\n${clean.blockingStatus}`;
		return;
	}
	try {
		await commitDirtyWorktree(worker);
		const workerHead = await git(worker.worktreePath, ["rev-parse", "HEAD"], 30_000);
		if (workerHead.code === 0) worker.commit = workerHead.stdout.trim();
		const baseHead = await git(run.cwd, ["rev-parse", "HEAD"], 30_000);
		if (worker.commit && worker.commit === baseHead.stdout.trim()) {
			worker.integrationStatus = "no_changes";
			worker.status = "integrated";
			return;
		}
		const rebase = await git(worker.worktreePath, ["rebase", run.baseBranch], 240_000);
		if (rebase.code !== 0) {
			await git(worker.worktreePath, ["rebase", "--abort"], 60_000);
			worker.status = "blocked";
			worker.integrationStatus = "conflict";
			worker.blocker = `Rebase onto ${run.baseBranch} failed:\n${rebase.stderr || rebase.stdout}`;
			return;
		}
		const merge = await git(run.cwd, ["merge", "--no-ff", worker.branch, "-m", `frontier: integrate ${worker.lane.title}`], 240_000);
		if (merge.code !== 0) {
			worker.status = "blocked";
			worker.integrationStatus = "failed";
			worker.blocker = `Merge failed:\n${merge.stderr || merge.stdout}`;
			return;
		}
		await runValidationIfRequested(run, worker);
		if (worker.status === "blocked") return;
		const integratedHead = await git(run.cwd, ["rev-parse", "HEAD"], 30_000);
		worker.integratedCommit = integratedHead.stdout.trim();
		worker.integrationStatus = "merged";
		worker.status = "integrated";
	} catch (error) {
		worker.status = "blocked";
		worker.integrationStatus = "failed";
		worker.blocker = error instanceof Error ? error.message : String(error);
	} finally {
		await saveRun(run);
	}
}

async function cleanupWorktree(run: FrontierRun, worker: WorkerState): Promise<void> {
	if (run.options.keepWorktrees || worker.status !== "integrated") return;
	if (await pathExists(worker.worktreePath)) await git(run.cwd, ["worktree", "remove", "--force", worker.worktreePath], 120_000);
}

async function runCurator(run: FrontierRun): Promise<void> {
	if (!run.options.curate) return;
	run.status = "curating";
	await saveRun(run);
	const curatorDir = path.join(run.rootDir, "curator");
	await fsp.mkdir(curatorDir, { recursive: true });
	const promptPath = path.join(curatorDir, "prompt.md");
	const sessionDir = path.join(curatorDir, "sessions");
	await fsp.mkdir(sessionDir, { recursive: true });
	await writeText(promptPath, makeCuratorPrompt(run));
	run.curatorPromptPath = promptPath;
	const result = await runPiJson(run.cwd, promptPath, sessionDir, run.options.curatorModel, run.config.workerEnv ?? {}, async (event) => {
		await fsp.appendFile(path.join(curatorDir, "events.jsonl"), JSON.stringify(event) + "\n");
	});
	run.curatorExitCode = result.exitCode;
	run.curatorFinalText = result.finalText;
	run.curatorStderr = result.stderr;
	await saveRun(run);

	const status = await porcelainStatus(run.cwd);
	const changed = status.split("\n").filter(Boolean).filter((line) => !isIgnorableDirtyLine(line));
	if (changed.length > 0) {
		await git(run.cwd, ["add", "wiki", ".frontier/lanes.json"], 60_000);
		const commit = await git(run.cwd, ["commit", "-m", `frontier: curate wiki ${run.id}`], 120_000);
		if (commit.code !== 0 && !/nothing to commit/i.test(commit.stderr + commit.stdout)) await appendEvent(run, { type: "curator_commit_failed", stderr: commit.stderr, stdout: commit.stdout });
	}
}

function safeSetFrontierStatus(value: string | undefined): void {
	try {
		currentCtx?.ui.setStatus("frontier", value);
	} catch {
		// Background frontier runs can outlive the command context that launched
		// them. State files/events are authoritative; stale UI handles should not
		// turn a completed run into a failed run.
	}
}

function safeNotifyFrontier(message: string, level: "info" | "warning" | "error" = "info"): void {
	try {
		currentCtx?.ui.notify(message, level);
	} catch {
		// See safeSetFrontierStatus.
	}
}

async function managerLoop(run: FrontierRun): Promise<void> {
	try {
		safeSetFrontierStatus(`frontier ${run.id}: running`);
		run.status = "running";
		await appendEvent(run, { type: "run_start", workers: run.workers.length });
		await saveRun(run);
		await runWorkersWithResources(run);
		if (run.status === "stopped") return;
		run.status = "integrating";
		await saveRun(run);
		safeSetFrontierStatus(`frontier ${run.id}: integrating`);
		for (const worker of run.workers) {
			if (run.status === "stopped") return;
			await integrateWorker(run, worker);
			await cleanupWorktree(run, worker);
		}
		if (run.status === "stopped") return;
		await runCurator(run);
		run.status = "completed";
		run.finishedAt = nowIso();
		await appendEvent(run, { type: "run_end", status: run.status });
		await saveRun(run);
		safeSetFrontierStatus(undefined);
		safeNotifyFrontier(`Frontier run ${run.id} completed. State: ${path.join(run.rootDir, "state.json")}`, "info");
	} catch (error) {
		run.status = "failed";
		run.finishedAt = nowIso();
		await appendEvent(run, { type: "run_failed", error: error instanceof Error ? error.stack ?? error.message : String(error) });
		await saveRun(run);
		safeSetFrontierStatus(undefined);
		safeNotifyFrontier(`Frontier run ${run.id} failed: ${error instanceof Error ? error.message : String(error)}`, "error");
	} finally {
		activeRun = undefined;
	}
}

function formatRunStatus(run: FrontierRun): string {
	const lines = [`Run ${run.id}: ${run.status}`, `Cwd: ${run.cwd}`, `State: ${path.join(run.rootDir, "state.json")}`, `Plan: ${run.managerPlanPath}`, ""];
	for (const worker of run.workers) {
		lines.push(
			`- ${worker.slug} ${worker.status}/${worker.integrationStatus} [${worker.lane.resource}] ${worker.lane.title}` +
				(worker.exitCode !== undefined ? ` exit=${worker.exitCode}` : "") +
				(worker.commit ? ` commit=${worker.commit.slice(0, 8)}` : "") +
				(worker.blocker ? `\n  blocker: ${worker.blocker.split("\n")[0]}` : "") +
				(worker.lastEvent ? `\n  last: ${worker.lastEvent}` : "") +
				(worker.lastText ? `\n  text: ${worker.lastText.replace(/\s+/g, " ").slice(-180)}` : ""),
		);
	}
	if (run.curatorExitCode !== undefined) lines.push(`\nCurator exit: ${run.curatorExitCode}`);
	return lines.join("\n");
}

async function latestRunFromDisk(cwd: string): Promise<FrontierRun | undefined> {
	const root = path.join(cwd, ".frontier", "runs");
	if (!(await pathExists(root))) return undefined;
	const entries = await fsp.readdir(root, { withFileTypes: true });
	const dirs = entries.filter((entry) => entry.isDirectory()).map((entry) => entry.name).sort().reverse();
	for (const dir of dirs) {
		const statePath = path.join(root, dir, "state.json");
		if (!(await pathExists(statePath))) continue;
		try {
			return JSON.parse(await readText(statePath)) as FrontierRun;
		} catch {
			continue;
		}
	}
	return undefined;
}

async function startFrontier(args: string, ctx: ExtensionCommandContext): Promise<void> {
	currentCtx = ctx;
	await ctx.waitForIdle();
	if (activeRun && !["completed", "failed", "stopped"].includes(activeRun.status)) {
		ctx.ui.notify(`A frontier run is already active: ${activeRun.id}`, "warning");
		return;
	}
	const config = await readConfig(ctx.cwd);
	const options = parseArgs(args, config);
	const clean = await assertBaseClean(ctx.cwd, options.allowDirty);
	if (!clean.clean) {
		ctx.ui.notify(`Base worktree is not clean. Commit/stash first or pass --allow-dirty.\n${clean.blockingStatus}`, "error");
		return;
	}
	const lanes = await readLanes(ctx.cwd);
	const selected = selectLanes(lanes, options);
	if (selected.length === 0) {
		ctx.ui.notify("No unblocked lanes matched the requested filters.", "warning");
		return;
	}
	const runId = compactRunId();
	const branch = config.baseBranch ?? (await currentBranch(ctx.cwd));
	const run: FrontierRun = {
		id: runId,
		cwd: ctx.cwd,
		baseBranch: branch,
		rootDir: runRoot(ctx.cwd, runId),
		worktreeRoot: worktreeRoot(ctx.cwd),
		startedAt: nowIso(),
		status: "planning",
		config,
		options,
		workers: [],
		managerPlanPath: path.join(runRoot(ctx.cwd, runId), "manager-plan.md"),
	};
	await fsp.mkdir(run.rootDir, { recursive: true });

	if (options.dryRun) {
		await writeText(path.join(run.rootDir, "dry-run.md"), `# Frontier dry run ${runId}\n\n${selected.map((lane) => `- ${lane.id}: ${lane.title} (${lane.heat}/${lane.resource}) — ${lane.nextAction}`).join("\n")}\n`);
		ctx.ui.notify(`Dry run selected ${selected.length} lane(s). See ${path.join(run.rootDir, "dry-run.md")}`, "info");
		return;
	}
	if (ctx.hasUI && !options.yes) {
		const ok = await ctx.ui.confirm("Start frontier run?", `Workers: ${selected.length}\nMerge: ${options.merge}\nCurator: ${options.curate}\n\n${selected.map((lane) => `- ${lane.title} [${lane.resource}]`).join("\n")}`);
		if (!ok) return;
	}
	await fsp.mkdir(run.worktreeRoot, { recursive: true });
	for (let i = 0; i < selected.length; i++) run.workers.push(await prepareWorker(run, selected[i], i));
	await writeManagerPlan(run);
	await saveRun(run);
	activeRun = run;
	ctx.ui.notify(`Frontier run ${run.id} started with ${run.workers.length} worker(s). State: ${path.join(run.rootDir, "state.json")}`, "info");
	void managerLoop(run);
}

async function showStatus(ctx: ExtensionCommandContext): Promise<void> {
	const run = activeRun ?? (await latestRunFromDisk(ctx.cwd));
	if (!run) {
		ctx.ui.notify("No frontier runs found.", "info");
		return;
	}
	const text = formatRunStatus(run);
	if (ctx.hasUI) await ctx.ui.editor("Frontier status", text);
	else console.log(text);
}

async function stopFrontier(ctx: ExtensionCommandContext): Promise<void> {
	if (!activeRun) {
		ctx.ui.notify("No active frontier run.", "info");
		return;
	}
	for (const proc of childProcesses.values()) {
		proc.kill("SIGTERM");
		setTimeout(() => {
			if (!proc.killed) proc.kill("SIGKILL");
		}, 5000).unref();
	}
	activeRun.status = "stopped";
	activeRun.finishedAt = nowIso();
	await appendEvent(activeRun, { type: "run_stopped" });
	await saveRun(activeRun);
	ctx.ui.setStatus("frontier", undefined);
	ctx.ui.notify(`Stopped frontier run ${activeRun.id}.`, "warning");
	activeRun = undefined;
}

async function curateOnly(args: string, ctx: ExtensionCommandContext): Promise<void> {
	currentCtx = ctx;
	await ctx.waitForIdle();
	const run = activeRun ?? (await latestRunFromDisk(ctx.cwd));
	if (!run) {
		ctx.ui.notify("No frontier run found to curate.", "warning");
		return;
	}
	const options = parseArgs(args, run.config);
	const clean = await assertBaseClean(ctx.cwd, options.allowDirty);
	if (!clean.clean) {
		ctx.ui.notify(`Base worktree is not clean. Commit/stash first or pass --allow-dirty.\n${clean.blockingStatus}`, "error");
		return;
	}
	run.options.curatorModel = options.curatorModel ?? run.options.curatorModel;
	run.options.curate = true;
	if (ctx.hasUI && !options.yes) {
		const ok = await ctx.ui.confirm("Run wiki curator?", `Run ${run.id}\nState: ${path.join(run.rootDir, "state.json")}`);
		if (!ok) return;
	}
	await runCurator(run);
	await saveRun(run);
	ctx.ui.notify(`Curator finished for run ${run.id}.`, "info");
}

async function initFrontier(args: string, ctx: ExtensionCommandContext): Promise<void> {
	const template = shellSplit(args)[0] || "ml-perf";
	await fsp.mkdir(path.join(ctx.cwd, ".frontier"), { recursive: true });
	await fsp.mkdir(path.join(ctx.cwd, "wiki", "ops", "open-notes"), { recursive: true });
	const created: string[] = [];
	const ensure = async (relativePath: string, content: string) => {
		const full = path.join(ctx.cwd, relativePath);
		if (await pathExists(full)) return;
		await writeText(full, content);
		created.push(relativePath);
	};
	await ensure(CONFIG_PATH, JSON.stringify({ template, maxWorkers: 5, resourceLimits: { gpu: 1, cpu: 3, code: 3, wiki: 1 }, defaultHeats: ["hot", "active", "warm"], wikiFiles: DEFAULT_WIKI_FILES }, null, 2) + "\n");
	await ensure(LANES_PATH, JSON.stringify([{ id: "baseline-harness", title: "Baseline harness", stage: "seeded", heat: "hot", resource: "code", goal: "Create a baseline receipt.", nextAction: "Inventory benchmark commands and record baseline surfaces." }], null, 2) + "\n");
	await ensure("wiki/ops/status.md", "# Frontier Status\n\nInitialized by `/frontier-init`.\n");
	await ensure("wiki/ops/frontier.md", "# Frontier\n\nSource of truth: `.frontier/lanes.json`.\n");
	await ensure("wiki/ops/experiment-ledger.md", "# Experiment Ledger\n\nNo receipts yet.\n");
	await ensure("wiki/ops/test-ledger.md", "# Test Ledger\n\nNo frontier tests yet.\n");
	ctx.ui.notify(created.length ? `Frontier initialized (${created.join(", ")}).` : "Frontier files already exist.", "info");
}

export default function (pi: ExtensionAPI) {
	pi.registerCommand("frontier-init", {
		description: "Initialize generic frontier lab files for this project. Use `ml-perf` for ML performance projects.",
		handler: async (args, ctx) => initFrontier(args, ctx),
	});
	pi.registerCommand("frontier-start", {
		description: "Start a BFS/hot frontier run with resource-aware fanout, git worktrees, receipts, merge gate, and wiki curator.",
		handler: async (args, ctx) => startFrontier(args, ctx),
	});
	pi.registerCommand("frontier-status", {
		description: "Show active or latest frontier run state.",
		handler: async (_args, ctx) => showStatus(ctx),
	});
	pi.registerCommand("frontier-stop", {
		description: "Stop active frontier run and terminate child pi processes.",
		handler: async (_args, ctx) => stopFrontier(ctx),
	});
	pi.registerCommand("frontier-curate", {
		description: "Run the wiki curator for the latest frontier run.",
		handler: async (args, ctx) => curateOnly(args, ctx),
	});
	pi.on("session_shutdown", async () => {
		for (const proc of childProcesses.values()) proc.kill("SIGTERM");
		childProcesses.clear();
	});
}
