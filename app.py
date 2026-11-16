"""
ChurnIQ - Customer Churn Prediction & Analytics Platform
Main Flask Application Entrypoint
"""

import logging
from flask import Flask, render_template, send_from_directory
from flask_cors import CORS

import config
from routes.prediction_routes import api_bp

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def create_app() -> Flask:
    """
    Creates and configures the Flask application instance.
    """
    app = Flask(
        __name__,
        template_folder=str(config.TEMPLATES_DIR),
        static_folder=str(config.STATIC_DIR),
    )

    # Configuration
    app.config["SECRET_KEY"] = config.SECRET_KEY
    app.config["DEBUG"] = config.FLASK_ENV == "development"

    # Enable Cross-Origin Resource Sharing
    CORS(app)

    # Register API blueprints
    app.register_blueprint(api_bp)

    # Web View Routes
    @app.route("/")
    def index_view():
        """Overview dashboard view."""
        return render_template("index.html")

    @app.route("/analytics")
    def analytics_view():
        """Exploratory and comparative analytics view."""
        return render_template("analytics.html")

    @app.route("/prediction")
    def prediction_view():
        """Customer churn prediction and risk evaluation view."""
        return render_template("prediction.html")

    @app.route("/reports/<path:filename>")
    def reports_file(filename):
        """Serves generated evaluation reports (e.g., confusion matrix image)."""
        return send_from_directory(str(config.REPORTS_DIR), filename)

    # Global Error Handlers
    @app.errorhandler(404)
    def not_found_error(error):
        return render_template("index.html"), 404

    @app.errorhandler(500)
    def internal_error(error):
        return render_template("index.html"), 500

    # Warm-up predictor if model and preprocessor are available
    if config.MODEL_PATH.exists() and config.PREPROCESSOR_PATH.exists():
        try:
            from src.prediction import get_prediction_engine
            get_prediction_engine()
            logger.info("ChurnPredictor engine pre-warmed successfully.")
        except Exception as e:
            logger.warning(f"Could not pre-warm ChurnPredictor: {e}")

    return app


app = create_app()

if __name__ == "__main__":
    logger.info(f"Starting ChurnIQ server on port {config.PORT} (Debug={app.config['DEBUG']})...")
    app.run(host="0.0.0.0", port=config.PORT, debug=app.config["DEBUG"])
