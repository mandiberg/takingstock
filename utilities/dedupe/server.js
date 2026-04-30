#!/usr/bin/env node
/**
 * Dedupe Viewer local server.
 * Usage: node server.js <root_dir> [port]
 *
 * Serves dedupe_viewer.html and provides three endpoints:
 *   GET  /api/pairs          → JSON array of all image pairs
 *   GET  /image?p=<relpath>  → serve an image file
 *   POST /api/delete-sql     → delete an SQL file  { "sqlPath": "<relpath>" }
 */

const http  = require('http');
const fs    = require('fs');
const path  = require('path');
const url   = require('url');

const rootArg = process.argv[2];
if (!rootArg) {
  console.error('Usage: node server.js <root_dir> [port]');
  process.exit(1);
}

const ROOT = path.resolve(rootArg);
const PORT = parseInt(process.argv[3] || '3000', 10);

if (!fs.existsSync(ROOT) || !fs.statSync(ROOT).isDirectory()) {
  console.error(`Error: '${ROOT}' is not a directory.`);
  process.exit(1);
}

// ── Directory crawl ─────────────────────────────────────────────────────────

function crawlPairs() {
  const pairs = [];
  for (const clusterName of sorted(fs.readdirSync(ROOT))) {
    if (clusterName.startsWith('.')) continue;
    const clusterPath = path.join(ROOT, clusterName);
    if (!fs.statSync(clusterPath).isDirectory()) continue;

    for (const tierName of sorted(fs.readdirSync(clusterPath))) {
      if (tierName.startsWith('.')) continue;
      const tierPath = path.join(clusterPath, tierName);
      if (!fs.statSync(tierPath).isDirectory()) continue;

      for (const scoreName of sorted(fs.readdirSync(tierPath))) {
        if (scoreName.startsWith('.')) continue;
        const scorePath = path.join(tierPath, scoreName);
        if (!fs.statSync(scorePath).isDirectory()) continue;

        const pair = collectPair(scorePath, clusterName, tierName, scoreName);
        if (pair) pairs.push(pair);
      }
    }
  }
  return pairs;
}

function collectPair(scorePath, clusterName, tierName, scoreName) {
  const entries = fs.readdirSync(scorePath);
  const jpgs = entries.filter(n => /\.(jpg|jpeg)$/i.test(n)).sort();
  const sql  = entries.find(n => /\.sql$/i.test(n)) || null;
  if (jpgs.length !== 2) return null;
  const rel = (name) => path.join(clusterName, tierName, scoreName, name);
  return {
    label:   `${clusterName}/${tierName}/${scoreName}`,
    imgA:    rel(jpgs[0]),
    imgB:    rel(jpgs[1]),
    sqlPath: sql ? rel(sql) : null,
  };
}

function sorted(arr) { return [...arr].sort(); }

// ── MIME types ───────────────────────────────────────────────────────────────

const MIME = { '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.html': 'text/html' };

// ── Request handler ──────────────────────────────────────────────────────────

const VIEWER_PATH = path.join(__dirname, 'dedupe_viewer.html');

const server = http.createServer((req, res) => {
  const parsed   = url.parse(req.url, true);
  const pathname = parsed.pathname;

  // CORS for local dev
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') { res.writeHead(204); res.end(); return; }

  // Serve the viewer page
  if (req.method === 'GET' && (pathname === '/' || pathname === '/index.html')) {
    serveFile(res, VIEWER_PATH, 'text/html');
    return;
  }

  // List all pairs
  if (req.method === 'GET' && pathname === '/api/pairs') {
    try {
      const pairs = crawlPairs();
      json(res, 200, pairs);
    } catch (e) {
      json(res, 500, { error: e.message });
    }
    return;
  }

  // Serve an image
  if (req.method === 'GET' && pathname === '/image') {
    const relPath = parsed.query.p;
    if (!relPath) { json(res, 400, { error: 'Missing ?p= parameter' }); return; }
    const absPath = path.resolve(ROOT, relPath);
    if (!absPath.startsWith(ROOT)) { json(res, 403, { error: 'Forbidden' }); return; }
    serveFile(res, absPath, MIME[path.extname(absPath).toLowerCase()] || 'application/octet-stream');
    return;
  }

  // Delete an SQL file
  if (req.method === 'POST' && pathname === '/api/delete-sql') {
    let body = '';
    req.on('data', chunk => { body += chunk; });
    req.on('end', () => {
      try {
        const { sqlPath } = JSON.parse(body);
        if (!sqlPath) { json(res, 400, { error: 'Missing sqlPath' }); return; }
        const absPath = path.resolve(ROOT, sqlPath);
        if (!absPath.startsWith(ROOT)) { json(res, 403, { error: 'Forbidden' }); return; }
        if (!absPath.endsWith('.sql')) { json(res, 400, { error: 'Not an SQL file' }); return; }
        if (fs.existsSync(absPath)) fs.unlinkSync(absPath);
        json(res, 200, { ok: true });
      } catch (e) {
        json(res, 500, { error: e.message });
      }
    });
    return;
  }

  json(res, 404, { error: 'Not found' });
});

function json(res, status, obj) {
  res.writeHead(status, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify(obj));
}

function serveFile(res, filePath, contentType) {
  fs.readFile(filePath, (err, data) => {
    if (err) { json(res, 404, { error: 'File not found' }); return; }
    res.writeHead(200, { 'Content-Type': contentType });
    res.end(data);
  });
}

server.listen(PORT, '127.0.0.1', () => {
  console.log(`Dedupe Viewer running at http://localhost:${PORT}`);
  console.log(`Root directory: ${ROOT}`);
});
