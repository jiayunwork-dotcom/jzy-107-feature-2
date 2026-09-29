"""应用入口：装配 Flask 应用、通道档仓库与路由。"""
from __future__ import annotations

from flask import Flask

from mhd.profiles import ProfileStore
from mhd.routes import bp, register_error_handlers


def create_app() -> Flask:
    app = Flask(__name__)
    app.json.ensure_ascii = False  # 错误原因等中文信息不转义，直接可读
    app.extensions["mhd_profile_store"] = ProfileStore()
    app.register_blueprint(bp)
    register_error_handlers(app)
    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
