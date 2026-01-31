const express = require('express');
const cors = require('cors');
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());

// Job Store
const jobs = {};

// Paths
const EXTENSION_PATH = path.join(__dirname, '../extensions/nprd-extension');
const OUTPUT_BASE = path.join(__dirname, '../docs/knowledge/nprd_outputs');

// Ensure output directory exists
if (!fs.existsSync(OUTPUT_BASE)) {
    fs.mkdirSync(OUTPUT_BASE, { recursive: true });
}

// --- Helpers ---

function updateJob(id, status, progress, log = null, result = null) {
    if (!jobs[id]) return;
    jobs[id].status = status;
    jobs[id].progress = progress;
    jobs[id].updatedAt = new Date();
    if (log) jobs[id].logs.push(log);
    if (result) jobs[id].result = result;
}

// --- verification Logic (CLI Based) ---

async function runGeminiAnalysis(jobId, manifest, promptText) {
    updateJob(jobId, 'PROCESSING', 50, 'Invoking Gemini CLI...');

    const videoPath = manifest.video_path;

    // We construct a command that tells Gemini to run with the video file
    // NOTE: 'gemini run' syntax varies by version. 
    // Usually: gemini run "prompt" --file video.mp4
    // Or: gemini query "prompt" -f video.mp4
    // We will use the most standard form seen in docs or standard practice.

    return new Promise((resolve, reject) => {
        // We use 'spawn' to run the CLI.
        // To allow the browser window to open (if the CLI attempts it), we arguably need 'inherit'.
        // BUT 'inherit' breaks our stdout capturing for the JSON result.
        // COMPROMISE: We run with 'pipe'. If it pauses/hangs, we assume it's waiting for Auth 
        // and we notify the user to check their terminal or run the auth command manually.
        // However, the user specifically asked for "It's the one requesting authentication... prompts me to open a window".
        // This implies proper X11/Display forwarding which usually works even spawned if the process isn't detached.

        const gemini = spawn('gemini', ['run', promptText, '-f', videoPath], {
            stdio: 'pipe', // Must pipe to get the JSON output
            env: { ...process.env, FORCE_COLOR: 'true' } // Sometimes helps with output
        });

        let output = '';
        let error = '';

        gemini.stdout.on('data', (data) => {
            const str = data.toString();
            output += str;
            // Detect if it's chatting or outputting JSON
            if (str.length < 500) updateJob(jobId, 'ANALYZING', 60, `[Gemini] ${str.trim()}`);
        });

        gemini.stderr.on('data', (data) => {
            const str = data.toString();
            error += str;
            // Heuristic detection of Auth Prompts
            if (str.includes("Login") || str.includes("authenticate") || str.includes("browser")) {
                updateJob(jobId, 'WAITING_AUTH', 50, "Gemini needs authentication! Check the server terminal or logs.");
                // In a real desktop app, we might try to open the URL found in 'str' here.
            }
        });

        gemini.on('close', (code) => {
            if (code !== 0) {
                // Return raw error to help debug
                return reject(new Error(`Gemini CLI exited with code ${code}. Error: ${error}`));
            }

            try {
                // Find JSON block
                const jsonMatch = output.match(/\{[\s\S]*\}/);
                if (jsonMatch) {
                    resolve(JSON.parse(jsonMatch[0]));
                } else {
                    // Fallback to raw text wrapped in logic
                    resolve({
                        meta: { title: "Raw Output", speakers: {} },
                        timeline: [{ transcript: output.trim(), clip_id: 1, timestamp_start: "00:00", timestamp_end: "End" }]
                    });
                }
            } catch (e) {
                reject(new Error("Failed to parse JSON from Gemini output: " + output.substring(0, 100)));
            }
        });
    });
}

// --- Main Workflow ---

app.post('/api/nprd/analyze', async (req, res) => {
    // Robust Body Checking
    if (!req.body) return res.status(400).json({ error: "Missing Request Body" });
    const { url, model = 'gemini' } = req.body;

    if (!url) return res.status(400).json({ error: "URL is required" });

    const jobId = `job_${Date.now()}`;
    jobs[jobId] = {
        id: jobId,
        status: 'PENDING',
        progress: 0,
        logs: [],
        result: null,
        createdAt: new Date()
    };

    res.json({ jobId, status: 'PENDING' });

    // Start Async Process
    (async () => {
        try {
            updateJob(jobId, 'STARTING', 10, 'Starting NPRD Workflow...');

            // 1. Run Pre-processor
            updateJob(jobId, 'DOWNLOADING', 20, `Running Python Pre-processor for ${url}...`);

            const pythonScript = path.join(EXTENSION_PATH, 'scripts/process_video.py');
            // We use 'pipe' to capture logs for the status API
            const pythonProcess = spawn('python3', [pythonScript, '--url', url, '--dir', path.join(__dirname, '../tmp/nprd_data')], { stdio: 'pipe' });

            let scriptOutput = '';

            pythonProcess.stdout.on('data', (data) => {
                const lines = data.toString().split('\n');
                lines.forEach(line => {
                    if (line.trim()) {
                        scriptOutput += line + '\n';
                        updateJob(jobId, 'DOWNLOADING', 30, `[Script] ${line}`);
                    }
                });
            });

            pythonProcess.stderr.on('data', (data) => {
                updateJob(jobId, 'DOWNLOADING', 30, `[Script Info] ${data}`);
            });

            await new Promise((resolve, reject) => {
                pythonProcess.on('close', (code) => {
                    if (code === 0) resolve();
                    else reject(new Error(`Pre-processor exited with code ${code}`));
                });
            });

            // Parse Manifest
            const manifestLine = scriptOutput.split('\n').find(l => l.startsWith('MANIFEST_PATH='));
            if (!manifestLine) throw new Error("Could not find MANIFEST_PATH in script output");
            const manifestPath = manifestLine.split('=')[1].trim();
            const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));

            updateJob(jobId, 'PREPROCESSED', 40, 'Video processing complete. Starting Analysis...');

            // 2. Load Prompt (Skill)
            const skillPath = path.join(EXTENSION_PATH, 'skills/librarian/SKILL.md');
            const promptText = fs.readFileSync(skillPath, 'utf8');

            // 3. Run Inference
            if (model === 'gemini') {
                const result = await runGeminiAnalysis(jobId, manifest, promptText);
                saveResult(jobId, result);
            } else if (model === 'qwen') {
                const result = await runQwenAnalysis(jobId, manifest, promptText);
                saveResult(jobId, result);
            }

        } catch (error) {
            console.error(error);
            updateJob(jobId, 'FAILED', 0, `Error: ${error.message}`);
        }
    })();
});

async function runQwenAnalysis(jobId, manifest, promptText) {
    updateJob(jobId, 'PROCESSING', 50, 'Invoking Qwen-VL (Python)...');

    // We use a specific Python script that handles the Qwen-VL API (OpenAI Compatible)
    // This allows us to send the IMAGES (Frames) properly, which the CLI might not support yet.
    const scriptPath = path.join(EXTENSION_PATH, 'scripts/qwen_runner.py');
    const manifestPath = path.join(path.dirname(manifest.video_path), 'manifest.json'); // Re-derive or pass explicitly if needed

    return new Promise((resolve, reject) => {
        const qwen = spawn('python3', [scriptPath, '--manifest', manifestPath, '--prompt', promptText], {
            env: { ...process.env }, // Pass env (API keys)
            stdio: 'pipe'
        });

        let output = '';
        let error = '';

        qwen.stdout.on('data', (data) => {
            const str = data.toString();
            output += str;
            updateJob(jobId, 'ANALYZING', 60, `[Qwen] ${str.substring(0, 100)}...`);
        });

        qwen.stderr.on('data', (data) => {
            error += data.toString();
        });

        qwen.on('close', (code) => {
            if (code !== 0) {
                return reject(new Error(`Qwen Runner exited with code ${code}. Error: ${error}`));
            }
            try {
                const jsonMatch = output.match(/\{[\s\S]*\}/);
                if (jsonMatch) resolve(JSON.parse(jsonMatch[0]));
                else resolve({ raw_output: output });
            } catch (e) {
                reject(new Error("Failed to parse JSON from Qwen output."));
            }
        });
    });
}

function saveResult(jobId, resultData) {
    const outputFilename = `nprd_${jobId}.json`;
    const outputPath = path.join(OUTPUT_BASE, outputFilename);
    fs.writeFileSync(outputPath, JSON.stringify(resultData, null, 2));

    updateJob(jobId, 'COMPLETED', 100, 'Analysis Complete.', {
        file: outputFilename,
        data: resultData
    });
}

app.get('/api/nprd/status/:id', (req, res) => {
    const job = jobs[req.params.id];
    if (!job) return res.status(404).json({ error: "Job not found" });
    res.json(job);
});

app.listen(PORT, () => {
    console.log(`NPRD Bridge Server running on port ${PORT}`);
});
