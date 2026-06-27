from __future__ import annotations

import os

import uvicorn


def main() -> int:
    uvicorn.run(
        "edl.main:app",
        host=os.getenv("EDL_HOST", "0.0.0.0"),
        port=int(os.getenv("EDL_PORT", "8200")),
        reload=os.getenv("EDL_RELOAD", "false").lower() == "true",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
