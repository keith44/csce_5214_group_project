from pathlib import Path
from prediction_monitor import init_db
import pandas as pd
from model_registry import get_model, list_models, activate_model
from prediction_monitor import init_db, log_prediction, get_predictions
from model_registry import register_model
from tempfile import TemporaryDirectory

from flask import (
    Flask,
    render_template,
    request,
    flash,
    redirect,
    url_for,
    get_flashed_messages,
)
import os
from time import perf_counter
from prediction_monitor import log_prediction

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "flight-project-local-development")
# initialize prediction monitor database
init_db()

# The available choices come from train.csv, not hardcoded airline/city lists.
training_data = pd.read_csv(
    BASE_DIR / "train.csv",
    usecols=["Source", "Destination", "Airline", "Total_Stops"],
).dropna()
training_data["stops"] = (
    training_data["Total_Stops"].str.extract(r"(\d+)")[0].fillna("0").astype(int)
)

flight_options = (
    training_data[["Source", "Destination", "Airline", "stops"]]
    .drop_duplicates()
    .sort_values(["Source", "Destination", "Airline", "stops"])
    .to_dict(orient="records")
)

sources = sorted(training_data["Source"].unique().tolist())
valid_flights = {
    (row["Source"], row["Destination"], row["Airline"], str(row["stops"]))
    for row in flight_options
}


def show_home(
    prediction_text="",
    error_text="",
    form_values=None,
    interval_text="",
    flight_summary=None,
    date_comparison=None,
    potential_savings="",
    date_comparison_airline="",
    airline_comparisons=None,
):

    form_values = form_values or {}
    airline_comparisons = airline_comparisons or []
    flight_cards = []

    # Original flight prediction
    if flight_summary:
        flight_cards.append(
            {
                **flight_summary,
                "title": "Flight Summary",
                "price": prediction_text,
                "interval": interval_text,
                "original": True,
            }
        )

    # Additional airline predictions
    for comparison in airline_comparisons:
        flight_cards.append(
            {
                **flight_summary,
                "title": "Airline Comparison",
                "airline": comparison["airline"],
                "price": comparison["price"],
                "interval": comparison["interval"],
                "original": False,
            }
        )

    # Find valid airlines that haven't already been selected.
    used_airlines = {flight["airline"] for flight in flight_cards}

    available_airlines = sorted(
        {
            row["Airline"]
            for row in flight_options
            if row["Source"] == form_values.get("Source")
            and row["Destination"] == form_values.get("Destination")
            and str(row["stops"]) == form_values.get("stops")
            and row["Airline"] not in used_airlines
        }
    )

    return render_template(
        "home.html",
        sources=sources,
        flight_options=flight_options,
        prediction_text=prediction_text,
        error_text=error_text,
        flight_cards=flight_cards,
        available_airlines=available_airlines,
        date_comparison=date_comparison,
        date_comparison_airline=date_comparison_airline,
        potential_savings=potential_savings,
        form_values=form_values,
    )


@app.route("/")
def home():
    results = get_flashed_messages(category_filter=["prediction"])

    if results:
        return show_home(**results[-1])

    return show_home()


@app.route("/predict", methods=["POST"])
def predict():
    form = request.form.to_dict()
    source = form.get("Source", "")
    destination = form.get("Destination", "")
    airline = form.get("airline", "")
    stops = form.get("stops", "")

    # Only offer/accept combinations found in training data.
    if (source, destination, airline, stops) not in valid_flights:
        return (
            show_home(
                error_text="That source, destination, airline, and stops combination is not in the training data.",
                form_values=form,
            ),
            400,
        )
    start = perf_counter()
    version = None
    try:
        departure = pd.to_datetime(form["Dep_Time"], format="%Y-%m-%dT%H:%M")
        arrival = pd.to_datetime(form["Arrival_Time"], format="%Y-%m-%dT%H:%M")
        duration_minutes = int((arrival - departure).total_seconds() / 60)
        if duration_minutes < 30:
            raise ValueError("Arrival must be at least 30 minutes after departure.")

        # Match the feature names and types in Flight_Price_Prediction_Student_Fixed.ipynb.
        # The saved pipeline handles categorical one-hot encoding automatically.
        flight = pd.DataFrame(
            [
                {
                    "Airline": airline,
                    "Source": source,
                    "Destination": destination,
                    "stops": int(stops),
                    "month": departure.month,
                    "day": departure.day,
                    "departure_hour": departure.hour,
                    "departure_minute": departure.minute,
                    "arrival_hour": arrival.hour,
                    "arrival_minute": arrival.minute,
                    "minutes": duration_minutes,
                }
            ]
        )

        start = perf_counter()

        model, metadata, version = get_model()

        predicted_fare = float(model.predict(flight)[0])

        interval = metadata["prediction_interval"]
        margin = interval["margin"]
        coverage = interval["coverage"]

        lower = max(0, predicted_fare - margin)
        upper = predicted_fare + margin

        response_time = (perf_counter() - start) * 1000

        log_prediction(version, predicted_fare, response_time, "success")

        flash(
            {
                "prediction_text": f"Rs. {predicted_fare:,.2f}",
                "interval_text": (
                    f"{coverage:.0%} prediction interval: "
                    f"Rs. {lower:,.2f} to Rs. {upper:,.2f}"
                ),
                "flight_summary": {
                    "source": source,
                    "destination": destination,
                    "airline": airline,
                    "stops": int(stops),
                    "departure": departure.strftime("%b %d, %Y %I:%M %p"),
                    "arrival": arrival.strftime("%b %d, %Y %I:%M %p"),
                    "duration": f"{duration_minutes // 60}h {duration_minutes % 60}m",
                },
                "form_values": form,
            },
            "prediction",
        )

        return redirect(url_for("home"), code=303)

    except (ValueError, KeyError) as error:
        return show_home(error_text=str(error), form_values=form), 400
    except FileNotFoundError:
        return (
            show_home(
                error_text="The active model file was not found. Check the model registry.",
                form_values=form,
            ),
            503,
        )
    except Exception as e:
        response_time = (perf_counter() - start) * 1000

        log_prediction(version, None, response_time, "error", str(e))

        app.logger.exception("Flight price prediction failed")

        return (
            show_home(
                error_text="Unable to generate a prediction. Please try again later.",
                form_values=form,
            ),
            500,
        )


@app.route("/compare-dates", methods=["POST"])
def compare_dates():
    form = request.form.to_dict()

    source = form.get("Source", "")
    destination = form.get("Destination", "")
    airline = form.get("airline", "")
    stops = form.get("stops", "")

    compare_airline = form.get("compare_airline", airline)
    selected_airlines = request.form.getlist("selected_airlines")
    all_airlines = [airline] + selected_airlines

    # Validate the requested comparison.
    if (
        len(selected_airlines) > 2
        or len(set(all_airlines)) != len(all_airlines)
        or compare_airline not in all_airlines
        or any(
            (source, destination, a, stops) not in valid_flights for a in all_airlines
        )
    ):
        return (
            show_home(
                error_text="Invalid airline or date comparison.", form_values=form
            ),
            400,
        )

    try:
        departure = pd.to_datetime(form["Dep_Time"], format="%Y-%m-%dT%H:%M")
        arrival = pd.to_datetime(form["Arrival_Time"], format="%Y-%m-%dT%H:%M")

        duration_minutes = int((arrival - departure).total_seconds() / 60)

        if duration_minutes < 30:
            raise ValueError("Arrival must be at least 30 minutes after departure.")

        def make_flight(a, dep, arr):
            return {
                "Airline": a,
                "Source": source,
                "Destination": destination,
                "stops": int(stops),
                "month": dep.month,
                "day": dep.day,
                "departure_hour": dep.hour,
                "departure_minute": dep.minute,
                "arrival_hour": arr.hour,
                "arrival_minute": arr.minute,
                "minutes": duration_minutes,
            }

        # Predictions for all currently displayed airline cards.
        flights = [make_flight(a, departure, arrival) for a in all_airlines]

        # Seven departure dates for the selected airline.
        for offset in range(-3, 4):
            dep = departure + pd.Timedelta(days=offset)
            arr = arrival + pd.Timedelta(days=offset)
            flights.append(make_flight(compare_airline, dep, arr))

        model, metadata, version = get_model()
        predictions = model.predict(pd.DataFrame(flights))

        card_prices = predictions[: len(all_airlines)]
        date_prices = predictions[len(all_airlines) :]

        interval = metadata["prediction_interval"]
        margin = interval["margin"]
        coverage = interval["coverage"]

        def format_interval(price):
            return (
                f"{coverage:.0%} prediction interval: "
                f"Rs. {max(0, price - margin):,.2f} "
                f"to Rs. {price + margin:,.2f}"
            )

        # Rebuild the additional airline cards.
        airline_comparisons = []

        for a, price in zip(selected_airlines, card_prices[1:]):
            airline_comparisons.append(
                {
                    "airline": a,
                    "price": f"Rs. {float(price):,.2f}",
                    "interval": format_interval(float(price)),
                }
            )

        # Build the seven-day comparison table.
        lowest_price = float(min(date_prices))
        selected_price = float(date_prices[3])
        savings = max(0, selected_price - lowest_price)

        comparison = []

        for offset, price in zip(range(-3, 4), date_prices):
            date = departure + pd.Timedelta(days=offset)

            comparison.append(
                {
                    "date": date.strftime("%b %d, %Y"),
                    "price": f"Rs. {float(price):,.2f}",
                    "selected": offset == 0,
                    "cheapest": float(price) == lowest_price,
                }
            )

        original_price = float(card_prices[0])

        flash(
            {
                "prediction_text": f"Rs. {original_price:,.2f}",
                "interval_text": format_interval(original_price),
                "flight_summary": {
                    "source": source,
                    "destination": destination,
                    "airline": airline,
                    "stops": int(stops),
                    "departure": departure.strftime("%b %d, %Y %I:%M %p"),
                    "arrival": arrival.strftime("%b %d, %Y %I:%M %p"),
                    "duration": f"{duration_minutes // 60}h {duration_minutes % 60}m",
                },
                "airline_comparisons": airline_comparisons,
                "date_comparison": comparison,
                "date_comparison_airline": compare_airline,
                "potential_savings": f"Rs. {savings:,.2f}",
                "form_values": {
                    **form,
                    "airline": airline,
                },
            },
            "prediction",
        )

        return redirect(url_for("home"), code=303)

    except (ValueError, KeyError) as error:
        return show_home(error_text=str(error), form_values=form), 400
    except Exception:
        app.logger.exception("Date comparison failed")
        return (
            show_home(
                error_text="Unable to compare departure dates.", form_values=form
            ),
            500,
        )


@app.route("/compare-airlines", methods=["POST"])
def compare_airlines():
    form = request.form.to_dict()
    source = form.get("Source", "")
    destination = form.get("Destination", "")
    airline = form.get("airline", "")
    stops = form.get("stops", "")

    selected_airlines = request.form.getlist("selected_airlines")
    new_airline = form.get("other_airline", "")
    selected_airlines.append(new_airline)

    if (
        len(selected_airlines) > 2
        or len(set([airline] + selected_airlines)) != len(selected_airlines) + 1
        or any(
            (source, destination, a, stops) not in valid_flights
            for a in [airline] + selected_airlines
        )
    ):
        return (
            show_home(error_text="Invalid airline comparison.", form_values=form),
            400,
        )

    try:
        departure = pd.to_datetime(form["Dep_Time"], format="%Y-%m-%dT%H:%M")
        arrival = pd.to_datetime(form["Arrival_Time"], format="%Y-%m-%dT%H:%M")
        duration_minutes = int((arrival - departure).total_seconds() / 60)

        if duration_minutes < 30:
            raise ValueError("Arrival must be at least 30 minutes after departure.")

        flights = []

        for a in [airline] + selected_airlines:
            flights.append(
                {
                    "Airline": a,
                    "Source": source,
                    "Destination": destination,
                    "stops": int(stops),
                    "month": departure.month,
                    "day": departure.day,
                    "departure_hour": departure.hour,
                    "departure_minute": departure.minute,
                    "arrival_hour": arrival.hour,
                    "arrival_minute": arrival.minute,
                    "minutes": duration_minutes,
                }
            )

        model, metadata, version = get_model()
        prices = model.predict(pd.DataFrame(flights))

        interval = metadata["prediction_interval"]
        margin = interval["margin"]
        coverage = interval["coverage"]

        def format_interval(price):
            return (
                f"{coverage:.0%} prediction interval: "
                f"Rs. {max(0, price - margin):,.2f} "
                f"to Rs. {price + margin:,.2f}"
            )

        comparisons = [
            {
                "airline": a,
                "price": f"Rs. {float(price):,.2f}",
                "interval": format_interval(float(price)),
            }
            for a, price in zip(selected_airlines, prices[1:])
        ]

        flash(
            {
                "prediction_text": f"Rs. {float(prices[0]):,.2f}",
                "interval_text": format_interval(float(prices[0])),
                "flight_summary": {
                    "source": source,
                    "destination": destination,
                    "airline": airline,
                    "stops": int(stops),
                    "departure": departure.strftime("%b %d, %Y %I:%M %p"),
                    "arrival": arrival.strftime("%b %d, %Y %I:%M %p"),
                    "duration": f"{duration_minutes // 60}h {duration_minutes % 60}m",
                },
                "airline_comparisons": comparisons,
                "form_values": form,
            },
            "prediction",
        )

        return redirect(url_for("home"), code=303)

    except (ValueError, KeyError) as error:
        return show_home(error_text=str(error), form_values=form), 400
    except Exception:
        app.logger.exception("Airline comparison failed")
        return (
            show_home(error_text="Unable to compare airlines.", form_values=form),
            500,
        )


@app.route("/admin/register", methods=["POST"])
def admin_register():
    version = request.form.get("version", "").strip()
    model_file = request.files.get("model_file")
    metadata_file = request.files.get("metadata_file")

    if not version or not model_file or not metadata_file:
        flash("Please provide a version, model file, and metadata file.", "admin")
        return redirect(url_for("admin"))

    if not version.startswith("v") or not version[1:].isdigit():
        flash("Model version must use the format v5, v6, etc.", "admin")
        return redirect(url_for("admin"))

    if not model_file.filename.endswith(".pkl") or not metadata_file.filename.endswith(
        ".json"
    ):
        flash("Please upload a .pkl model and a .json metadata file.", "admin")
        return redirect(url_for("admin"))

    try:
        with TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "model.pkl"
            metadata_path = Path(temp_dir) / "model_metadata.json"

            model_file.save(model_path)
            metadata_file.save(metadata_path)

            register_model(model_path, version)

        flash(f"Model {version} registered successfully.", "admin")

    except (FileExistsError, FileNotFoundError, ValueError) as e:
        flash(str(e), "admin")

    return redirect(url_for("admin"))


@app.route("/admin/activate/<version>", methods=["POST"])
def admin_activate(version):
    try:
        activate_model(version)
        flash(f"Model {version} activated successfully.", "admin")
    except FileNotFoundError as e:
        flash(str(e), "admin")

    return redirect(url_for("admin"))


if __name__ == "__main__":
    app.run(debug=True)
