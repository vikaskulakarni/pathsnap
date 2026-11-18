"""Evaluator interface for the PathSnap API running on Laptop 1."""

import os
from math import isfinite
from typing import Any
from urllib.parse import urlsplit

import folium
import requests
import streamlit as st
from streamlit_folium import st_folium

REQUEST_TIMEOUT = 30
DEFAULT_API_URL = "http://localhost:8000"


class APIError(Exception):
    """An API failure that can be shown directly in the UI."""


def api_request(base_url: str, route: str, *, trip_id: str | None = None) -> Any:
    """Use the same timeout and error handling across all three endpoints."""
    try:
        parsed = urlsplit(base_url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError
        parsed.port  # Validate an explicitly provided port.
    except ValueError as exc:
        raise APIError("Enter a valid HTTP or HTTPS API address.") from exc

    method = "GET" if trip_id is None else "POST"
    payload = None if trip_id is None else {"trip_id": trip_id}
    try:
        response = requests.request(
            method, f"{base_url}{route}", json=payload, timeout=REQUEST_TIMEOUT,
        )
    except requests.Timeout as exc:
        raise APIError("Laptop 1 took more than 30 seconds to respond. Try again.") from exc
    except requests.ConnectionError as exc:
        raise APIError(
            "Could not reach Laptop 1. Check the API address and that both laptops are on the same network."
        ) from exc
    except requests.RequestException as exc:
        raise APIError("The API request failed. Check the API address and try again.") from exc
    if response.status_code != 200:
        raise APIError(f"Laptop 1 returned HTTP {response.status_code} for {route}.")
    try:
        return response.json()
    except ValueError as exc:
        raise APIError(f"Laptop 1 returned invalid JSON for {route}.") from exc


def map_position(lon: float, lat: float) -> tuple[float, float]:
    """Convert the API's longitude-first coordinates into Folium positions."""
    if not (-180 <= lon <= 180 and -90 <= lat <= 90):
        raise ValueError("Invalid coordinates")
    return lat, lon


def prediction_map(result: dict[str, Any]) -> folium.Map:
    """Draw only the geometry returned by the API."""
    gps = [map_position(point["lon"], point["lat"]) for point in result["points"]]
    route = [map_position(lon, lat) for lon, lat in result["route_coords"]]
    bounds = gps + route
    route_map = folium.Map(
        location=bounds[0] if bounds else None,
        zoom_start=15 if bounds else 2,
        prefer_canvas=True,
    )
    gps_layer = folium.FeatureGroup(name="GPS observations").add_to(route_map)
    for position in gps:
        folium.CircleMarker(
            position, radius=3, color="#2563eb", fill=True, fill_opacity=0.8,
        ).add_to(gps_layer)
    route_layer = folium.FeatureGroup(name="Predicted route").add_to(route_map)
    if len(route) >= 2:
        folium.PolyLine(route, color="#dc2626", weight=4).add_to(route_layer)
    elif route:
        folium.CircleMarker(
            route[0], radius=4, color="#dc2626", fill=True, fill_opacity=0.9,
        ).add_to(route_layer)
    if bounds:
        route_map.fit_bounds(bounds, padding=(20, 20), max_zoom=17)
    folium.LayerControl().add_to(route_map)
    route_map.get_root().html.add_child(folium.Element(
        '<div style="position:fixed;bottom:24px;left:12px;z-index:1000;'
        'background:white;color:#172033;padding:8px 12px;border:1px solid #d1d5db;'
        'border-radius:4px;font-size:13px">'
        '<div><span style="color:#2563eb">&#9679;</span> GPS observations</div>'
        '<div><span style="color:#dc2626">&#9473;</span> Predicted route</div></div>'
    ))
    return route_map


def percentage(value: float | None) -> str:
    if value is None:
        return "Unavailable"
    if not 0 <= value <= 1:
        raise ValueError("Expected a score between zero and one.")
    return f"{value:.1%}"


def read_prediction(response: Any, trip_id: str) -> dict[str, Any]:
    """Check the fields needed to render the selected trip."""
    if not isinstance(response, dict) or response.get("trip_id") != trip_id:
        raise APIError("Laptop 1 returned a prediction for a different trip.")
    try:
        points = response["points"]
        if type(response["n_points"]) is not int or response["n_points"] != len(points):
            raise ValueError("Point count does not match")
        if not isinstance(points, list) or not isinstance(response["route_coords"], list):
            raise ValueError("Expected coordinate lists")
        if not isfinite(response["latency_ms"]) or response["latency_ms"] < 0:
            raise ValueError("Invalid latency")
        if not isinstance(response["model_version"], str):
            raise ValueError("Invalid model version")
        percentage(response["accuracy"])
        percentage(response["route_continuity"])
        for point in points:
            map_position(point["lon"], point["lat"])
        for lon, lat in response["route_coords"]:
            map_position(lon, lat)
    except (KeyError, TypeError, ValueError) as exc:
        raise APIError("Laptop 1 returned a prediction in an unexpected format.") from exc
    return response


def main() -> None:
    st.set_page_config(page_title="PathSnap", layout="wide")
    st.title("PathSnap")
    with st.sidebar:
        api_url = st.text_input(
            "Laptop 1 API URL", value=os.getenv("PATHSNAP_API_URL", DEFAULT_API_URL),
        ).strip().rstrip("/")
        if st.button("Check connection"):
            try:
                health = api_request(api_url, "/health")
                if health != {"status": "ok"}:
                    raise APIError("The API responded, but did not report that it is healthy.")
                st.success("Connected to Laptop 1.")
            except APIError as exc:
                st.error(str(exc))
        refresh = st.button("Refresh trips")

    if refresh or st.session_state.get("catalog_url") != api_url:
        st.session_state.pop("catalog_url", None)
        try:
            trips = api_request(api_url, "/trips")
            if not isinstance(trips, list) or any(
                not isinstance(trip, str) or not trip for trip in trips
            ):
                raise APIError("Laptop 1 returned an invalid trip list.")
            st.session_state.update(catalog_url=api_url, trips=trips)
        except APIError as exc:
            st.error(str(exc))
            return
    trips = st.session_state["trips"]
    if not trips:
        st.info("Laptop 1 has no trips available.")
        return

    trip_id = st.selectbox("Trip", trips)
    if st.button("Predict", type="primary"):
        try:
            with st.spinner(f"Matching trip {trip_id} to the road network…"):
                result = read_prediction(api_request(api_url, "/predict", trip_id=trip_id), trip_id)
            st.session_state["last_prediction"] = (api_url, trip_id, result)
        except APIError as exc:
            st.error(str(exc))

    last = st.session_state.get("last_prediction")
    if last is None or last[:2] != (api_url, trip_id):
        return

    result = last[2]
    accuracy, continuity, points, latency = st.columns(4)
    accuracy.metric("Accuracy", percentage(result["accuracy"]))
    continuity.metric("Route continuity", percentage(result["route_continuity"]))
    points.metric("GPS points", f"{result['n_points']:,}")
    latency.metric("API latency", f"{result['latency_ms']:.0f} ms")
    st.text(f"Model version · {result['model_version']}")
    if result["points"] or result["route_coords"]:
        st_folium(prediction_map(result), width=None, height=520, returned_objects=[])
    else:
        st.info("The API returned no coordinates to map.")
    with st.expander("API response"):
        st.json(result, expanded=False)


if __name__ == "__main__":
    main()
