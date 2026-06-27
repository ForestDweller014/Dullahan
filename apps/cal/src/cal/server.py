from __future__ import annotations

import os

import uvicorn


def main() -> int:
    uvicorn.run(
        "cal.main:app",
        host=os.getenv("CAL_HOST", "0.0.0.0"),
        port=int(os.getenv("CAL_PORT", "8100")),
        reload=os.getenv("CAL_RELOAD", "false").lower() == "true",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
