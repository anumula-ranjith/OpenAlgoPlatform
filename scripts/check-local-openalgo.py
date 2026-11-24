"""Read-only local installation checks; never sends broker requests or orders."""

import asyncio
import importlib.metadata
import json
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from urllib.request import ProxyHandler, Request, build_opener
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
results = []


def check(name, operation):
    try:
        detail = operation()
        results.append({"check": name, "status": "PASS", "detail": detail})
    except Exception as error:
        results.append({"check": name, "status": "FAIL", "detail": str(error)})


def config_check():
    from dotenv import dotenv_values

    config = dotenv_values(ROOT / ".env")
    sample = dotenv_values(ROOT / ".sample.env")
    assert config["ENV_CONFIG_VERSION"] == sample["ENV_CONFIG_VERSION"], "Configuration version differs from template"
    for key in ("APP_KEY", "API_KEY_PEPPER", "FERNET_SALT"):
        value = config.get(key, "")
        assert len(value) >= 32 and value != sample[key], f"{key} is missing or still a placeholder"
    assert config["FLASK_HOST_IP"] == "127.0.0.1", "Web server is not bound to loopback"
    assert config["WEBSOCKET_HOST"] == "127.0.0.1", "WebSocket is not bound to loopback"
    assert config["ZMQ_HOST"] == "127.0.0.1", "ZeroMQ is not bound to loopback"
    assert config["FLASK_DEBUG"].lower() == "false", "Debug mode is enabled"
    assert "angel" in config["VALID_BROKERS"].split(","), "Angel One plugin is disabled"
    return "Matching configuration; unique installation secrets; loopback interfaces; debug disabled; Angel One enabled"


check("configuration", config_check)

from dotenv import dotenv_values

config = dotenv_values(ROOT / ".env")
base_url = f"http://127.0.0.1:{config.get('FLASK_PORT', '5000')}"
opener = build_opener(ProxyHandler({}))


def get(path):
    with opener.open(Request(base_url + path), timeout=15) as response:
        assert response.status == 200, f"HTTP {response.status}"
        return response.read()


def dependency_check():
    names = ("Flask", "SQLAlchemy", "openalgo", "websockets", "duckdb")
    return {name: importlib.metadata.version(name) for name in names}


def frontend_check():
    html = get("/setup").decode("utf-8")
    assert 'id="root"' in html, "React application root is missing"
    assets = re.findall(r'(?:src|href)="(/assets/[^\"]+\.(?:js|css))"', html)
    assert assets, "No bundled JavaScript/CSS assets found"
    for asset in assets:
        assert get(asset), f"Empty asset: {asset}"
    return f"Setup page and {len(assets)} referenced JavaScript/CSS assets served successfully"


def health_check():
    health = json.loads(get("/health/check"))
    assert health["serviceId"] == "openalgo", "Unexpected service at configured URL"
    assert health["status"] in ("pass", "warn"), "Application health failed"
    databases = health["checks"].get("database:connectivity", [])
    assert databases, "No database health results returned"
    assert all(item["status"] == "pass" for item in databases), "A database is unhealthy"
    return {"overall": health["status"], "databases": {item["componentId"]: item["status"] for item in databases}}


def database_check():
    database_path = ROOT / "db" / "openalgo.db"
    with sqlite3.connect(database_path.as_uri() + "?mode=ro", uri=True) as connection:
        assert connection.execute("PRAGMA quick_check").fetchone()[0] == "ok", "Database integrity failed"
        mode = connection.execute("SELECT analyze_mode FROM settings LIMIT 1").fetchone()
        assert mode is not None, "Execution mode has not been initialized"
        users = connection.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    return {"integrity": "ok", "execution_mode": "Analyze" if mode[0] else "Live", "administrator_accounts": users}


async def websocket_check():
    from websockets.asyncio.client import connect

    url = config["WEBSOCKET_URL"]
    async with connect(url, open_timeout=10, proxy=None):
        return f"WebSocket handshake succeeded at {url} (no authentication or market-data subscription attempted)"


check("dependencies", dependency_check)
check("frontend", frontend_check)
check("health_and_database_connectivity", health_check)
check("database_integrity_and_mode", database_check)
check("websocket_handshake", lambda: asyncio.run(websocket_check()))
print(json.dumps({
    "verified_at_ist": datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(),
    "url": base_url,
    "checks": results,
    "not_verified": ["Broker authentication", "Live market data", "Order placement", "Strategy execution"],
}, indent=2))
sys.exit(1 if any(result["status"] == "FAIL" for result in results) else 0)
