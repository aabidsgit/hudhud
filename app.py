# app.py
# -----------------------------------------------------------------------------
# HudHud Riyadh AI Assistant — Streamlit app (OpenAI GPT-5, streaming enabled)
#
# Quickstart:
#   pip install openai streamlit python-dotenv
#   # mac/linux: export OPENAI_API_KEY="your_api_key"
#   # windows powershell: setx OPENAI_API_KEY "your_api_key"
#   # (optional) create a .env with OPENAI_API_KEY and we'll load it automatically
#   streamlit run app.py
#
# Notes:
# - Streaming is ON by default. If streaming ever fails,
#   we gracefully fall back to non-streaming.
# - Some GPT-5 models fix temperature=1; we only send temperature when supported.
# - Output sanitizer removes raw tool-call stubs like "[google_maps_search]".
# - Includes 7 CEO-ready demo prompts as sidebar buttons.
# - Mock “tools” return deterministic demo data — swap with HudHud APIs later.
# -----------------------------------------------------------------------------

import os
import json
import math
import re
from typing import List, Dict, Any, Optional

import streamlit as st
from dotenv import load_dotenv, find_dotenv
import openai
from openai import OpenAI

# -----------------------------------------------------------------------------
# Robust .env loading (make .env override OS env if both exist)
# -----------------------------------------------------------------------------
# If you prefer OS env to win, set override=False.
load_dotenv(find_dotenv(usecwd=True), override=True)


# ---------------------------
# Geospatial helpers (demo)
# ---------------------------

def haversine_miles(lat1, lon1, lat2, lon2):
    """Great-circle distance between two points (miles)."""
    R_km = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return (R_km * c) * 0.621371  # km -> miles


# ---------------------------
# MOCK DATA (DEMO ONLY)
# Replace these with HudHud APIs later
# ---------------------------

RIYADH_CENTER = (24.7136, 46.6753)
OLAYA = (24.6920, 46.6850)        # approximate Olaya
DIRIYAH = (24.7370, 46.5750)      # approximate Diriyah

MOCK_MALLS = [
    {
        "name": "Riyadh Park Mall",
        "lat": 24.7555,
        "lon": 46.6422,
        "address": "Riyadh Park, Northern Ring Rd, Riyadh",
        "brands": ["Coach", "Apple", "Zara", "H&M"],
    },
    {
        "name": "Al Nakheel Mall",
        "lat": 24.7745,
        "lon": 46.7410,
        "address": "Exit 9, Al Imam Saud Ibn Abdulaziz Rd, Riyadh",
        "brands": ["Apple", "Sephora", "H&M", "Zara"],
    },
    {
        "name": "Kingdom Centre",
        "lat": 24.7117,
        "lon": 46.6740,
        "address": "Olaya St, Riyadh",
        "brands": ["Coach", "Dior", "Louis Vuitton"],
    },
]

MOCK_RESTAURANTS = [
    {
        "name": "Spice Route Indian Kitchen",
        "lat": 24.7532, "lon": 46.6489,
        "address": "Near Riyadh Park, Northern Ring Rd, Riyadh",
        "cuisine": "Indian", "open_late": True, "family_friendly": True, "mall": None
    },
    {
        "name": "Masala Junction",
        "lat": 24.7728, "lon": 46.7417,
        "address": "Near Al Nakheel Mall, Riyadh",
        "cuisine": "Indian", "open_late": False, "family_friendly": True, "mall": None
    },
    {
        "name": "Royal Biryani House",
        "lat": 24.6999, "lon": 46.6801,
        "address": "Olaya, Riyadh",
        "cuisine": "Indian", "open_late": True, "family_friendly": True, "mall": None
    },
    # Italian near/inside Riyadh Park Mall for demo
    {
        "name": "Trattoria Riyadh Park",
        "lat": 24.7558, "lon": 46.6425,
        "address": "Inside Riyadh Park Mall",
        "cuisine": "Italian", "open_late": True, "family_friendly": True, "mall": "Riyadh Park Mall"
    },
    {
        "name": "La Famiglia Olaya",
        "lat": 24.7040, "lon": 46.6800,
        "address": "Olaya St, Riyadh",
        "cuisine": "Italian", "open_late": True, "family_friendly": True, "mall": None
    },
    {
        "name": "Nonna’s Kitchen",
        "lat": 24.7110, "lon": 46.6690,
        "address": "Near Kingdom Centre, Olaya",
        "cuisine": "Italian", "open_late": False, "family_friendly": True, "mall": None
    },
    # Family-friendly inside Riyadh Park (non-Italian)
    {
        "name": "Garden Grill (Family Zone)",
        "lat": 24.7556, "lon": 46.6420,
        "address": "Food Court, Riyadh Park Mall",
        "cuisine": "International", "open_late": True, "family_friendly": True, "mall": "Riyadh Park Mall"
    },
]

MOCK_CAFES = [
    {
        "name": "Skyline Rooftop Café",
        "lat": 24.7055, "lon": 46.6845,
        "address": "Olaya, Riyadh",
        "rooftop": True, "open_after_midnight": True
    },
    {
        "name": "Vista Lounge",
        "lat": 24.7300, "lon": 46.6550,
        "address": "Northern Ring Rd, Riyadh",
        "rooftop": True, "open_after_midnight": False
    },
]

MOCK_CINEMAS = [
    {
        "name": "VOX Cinemas (IMAX) - Olaya",
        "lat": 24.7030, "lon": 46.6825,
        "address": "Olaya District, Riyadh",
        "imax": True
    },
    {
        "name": "Muvi Cinemas - Riyadh Park",
        "lat": 24.7552, "lon": 46.6428,
        "address": "Riyadh Park Mall",
        "imax": False
    },
]

MOCK_PETROL = [
    {
        "name": "QuickFuel + Wash",
        "lat": 24.7200, "lon": 46.6400,
        "address": "Northern Ring Rd, Riyadh",
        "car_wash": True
    },
    {
        "name": "Diriyah Fuel & Shine",
        "lat": 24.7305, "lon": 46.6000,
        "address": "Route toward Diriyah",
        "car_wash": True
    },
    {
        "name": "Central Fuel Stop",
        "lat": 24.7005, "lon": 46.6600,
        "address": "Olaya Connector",
        "car_wash": False
    },
]


# ---------------------------
# MOCK TOOL FUNCTIONS (DEMO)
# Replace with HudHud APIs later
# ---------------------------

def mock_find_brand_in_malls(brand: str, near_lat: float, near_lon: float, radius_miles: float):
    results = []
    for m in MOCK_MALLS:
        if brand.lower() in (b.lower() for b in m["brands"]):
            dist = haversine_miles(near_lat, near_lon, m["lat"], m["lon"])
            if dist <= radius_miles:
                results.append({
                    "mall": m["name"],
                    "address": m["address"],
                    "distance_miles": round(dist, 2),
                    "brands": m["brands"],
                })
    results.sort(key=lambda x: x["distance_miles"])
    return results


def mock_find_malls_with_brands(brands: List[str], near_lat: float, near_lon: float, radius_miles: float):
    brands_lower = [b.lower() for b in brands]
    results = []
    for m in MOCK_MALLS:
        mall_brands = [b.lower() for b in m["brands"]]
        if all(b in mall_brands for b in brands_lower):
            dist = haversine_miles(near_lat, near_lon, m["lat"], m["lon"])
            if dist <= radius_miles:
                results.append({
                    "mall": m["name"],
                    "address": m["address"],
                    "distance_miles": round(dist, 2),
                    "brands": m["brands"],
                })
    results.sort(key=lambda x: x["distance_miles"])
    return results


def mock_search_places(query: str, near_lat: float, near_lon: float, radius_miles: float, filters: Optional[Dict[str, Any]] = None):
    filters = filters or {}
    cuisine = filters.get("cuisine")
    open_late = filters.get("open_late")
    family = filters.get("family_friendly")
    mall = filters.get("mall")

    results = []
    for r in MOCK_RESTAURANTS:
        if cuisine and r["cuisine"].lower() != cuisine.lower():
            continue
        if open_late is not None and r["open_late"] != open_late:
            continue
        if family is not None and r["family_friendly"] != family:
            continue
        if mall and (r["mall"] or "").lower() != mall.lower():
            continue
        dist = haversine_miles(near_lat, near_lon, r["lat"], r["lon"])
        if dist <= radius_miles:
            results.append({
                "name": r["name"],
                "address": r["address"],
                "distance_miles": round(dist, 2),
                "cuisine": r["cuisine"],
                "open_late": r["open_late"],
                "family_friendly": r["family_friendly"],
                "mall": r["mall"],
            })
    results.sort(key=lambda x: x["distance_miles"])
    return results


def mock_search_rooftop_cafes(open_after_midnight: bool, near_lat: float, near_lon: float, radius_miles: float):
    results = []
    for c in MOCK_CAFES:
        if open_after_midnight and not c["open_after_midnight"]:
            continue
        dist = haversine_miles(near_lat, near_lon, c["lat"], c["lon"])
        if dist <= radius_miles:
            results.append({
                "name": c["name"],
                "address": c["address"],
                "distance_miles": round(dist, 2),
                "rooftop": c["rooftop"],
                "open_after_midnight": c["open_after_midnight"],
            })
    results.sort(key=lambda x: x["distance_miles"])
    return results


def mock_find_imax_near(lat_ref: float, lon_ref: float, radius_miles: float):
    results = []
    for c in MOCK_CINEMAS:
        if not c["imax"]:
            continue
        dist = haversine_miles(lat_ref, lon_ref, c["lat"], c["lon"])
        if dist <= radius_miles:
            results.append({
                "name": c["name"],
                "address": c["address"],
                "distance_miles": round(dist, 2),
                "imax": True
            })
    results.sort(key=lambda x: x["distance_miles"])
    return results


def mock_petrol_with_carwash_on_way_to_diriyah(near_lat: float, near_lon: float, radius_miles: float):
    """Very simple 'on the way' logic: points roughly west/northwest from user toward Diriyah and within radius."""
    results = []
    for p in MOCK_PETROL:
        if not p["car_wash"]:
            continue
        dist = haversine_miles(near_lat, near_lon, p["lat"], p["lon"])
        # crude directional filter toward Diriyah
        toward_diriyah = (p["lon"] <= near_lon + 0.02)  # west-ish heuristic
        if dist <= radius_miles and toward_diriyah:
            results.append({
                "name": p["name"],
                "address": p["address"],
                "distance_miles": round(dist, 2),
                "car_wash": True
            })
    results.sort(key=lambda x: x["distance_miles"])
    return results


# ---------------------------
# OpenAI client + chat logic
# ---------------------------

def _mask_key(k: Optional[str]) -> str:
    if not k:
        return "(missing)"
    if len(k) <= 12:
        return "****" + k[-4:]
    return k[:6] + "…" + k[-6:]


def get_openai_client() -> OpenAI:
    """
    Resolve the API key with clear, safe diagnostics.
    Precedence (adjust if desired):
      1) Environment variable (after .env override)
      2) streamlit secrets (if present and env missing)
    """
    key = os.getenv("OPENAI_API_KEY")
    source = "env"

    # If not found in env, allow Streamlit secrets to supply it
    try:
        if (not key) and ("OPENAI_API_KEY" in st.secrets):
            key = st.secrets["OPENAI_API_KEY"]
            source = "st.secrets"
    except Exception:
        # st.secrets might not be available outside Streamlit runtime
        pass
    if not key:
        st.error("OPENAI_API_KEY not found. Set it in .env, OS env, or .streamlit/secrets.toml.")
        st.stop()
    return OpenAI(api_key=key)


SYSTEM_POLICIES = """
You are HudHud Maps’ AI Assistant for Riyadh only.

Answer FORMAT rules (very important):
- Respond in natural, user-facing text (sentences/bullets). DO NOT output raw tool calls, labels, or pseudo-code such as:
  [google_maps_search], [tool:], "query: ...", JSON blobs, or code-fenced blocks for tools.
- If you lack live data or TOOL_RESULTS, say what info you need (e.g., neighborhood, distance) or explain the limitation briefly.
- When TOOL_RESULTS are present, ONLY use those results—do not invent venues or brands.

Other policies:
- Coverage is limited to **Riyadh**. If asked about outside areas, explain coverage is Riyadh-only and suggest Riyadh alternatives.
- Prefer concise, actionable answers with addresses and approximate distances.
- If the user’s request lacks a location, ask to use current location or a landmark/neighborhood (Olaya, Diriyah, KKIA).
- If a radius is missing for proximity searches, default to 10 miles.
- Be careful with prayer times; if unsure, suggest confirming at the venue.
- Ask clarifying questions when needed (cuisine, price, hours) before recommending.
"""


def build_messages(history: List[Dict[str, str]], user_message: str, tool_context: Optional[str]):
    preamble = (
        SYSTEM_POLICIES.strip()
        + "\n\nCurrent user’s assumed city: Riyadh, Saudi Arabia.\n"
        + "If 'TOOL_RESULTS' is present below, you MUST ground your answer only in that data and still speak in natural text.\n"
    )

    messages = [{"role": "system", "content": preamble}]

    for msg in history:
        role = "user" if msg["role"] == "user" else "assistant"
        messages.append({"role": role, "content": msg["content"]})

    user_text = user_message
    if tool_context:
        user_text += "\n\nTOOL_RESULTS:\n" + tool_context

    messages.append({"role": "user", "content": user_text})
    return messages


def model_supports_temperature(model: str) -> bool:
    # Heuristic: many GPT-5 class models fix temperature=1; older models allow it.
    return not model.startswith("gpt-5")


def stream_openai_response(
    client: OpenAI,
    model: str,
    messages,
    temperature: float,
    want_stream: bool = True,
):
    """
    Streams if allowed; otherwise falls back to non-streaming automatically.
    Yields text deltas (or a single final chunk when non-streaming).
    """
    kwargs = {"model": model, "messages": messages}
    if model_supports_temperature(model):
        kwargs["temperature"] = temperature

    # Try streaming if requested
    if want_stream:
        try:
            stream = client.chat.completions.create(stream=True, **kwargs)
            for chunk in stream:
                delta = getattr(chunk.choices[0].delta, "content", None)
                if delta:
                    yield delta
            return
        except openai.BadRequestError:
            # If streaming is somehow not allowed in the moment, fall back
            pass

    # Non-streaming fallback
    resp = client.chat.completions.create(**kwargs)
    final_text = resp.choices[0].message.content or ""
    yield final_text


# ---------------------------
# Output sanitizer to strip tool-call stubs
# ---------------------------

TOOLY_HEAD_RE = re.compile(r"^\s*\[[^\]]+\]\s*(?:\n|$)", flags=re.IGNORECASE | re.MULTILINE)
QUERY_LINE_RE = re.compile(r"^\s*query\s*:\s*.*$", flags=re.IGNORECASE | re.MULTILINE)
FENCED_CODE_RE = re.compile(r"```[\s\S]*?```", flags=re.MULTILINE)

def sanitize_model_output(text: str) -> str:
    text = FENCED_CODE_RE.sub("", text)
    text = TOOLY_HEAD_RE.sub("", text)
    text = QUERY_LINE_RE.sub("", text)
    text = re.sub(r"\[[a-z0-9_\-]+(?:\:[^\]]+)?\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


# ---------------------------
# Intent router (demo)
# ---------------------------

def extract_radius_miles(q: str, default_radius_miles: float) -> float:
    m = re.search(r"within\s+(\d+(\.\d+)?)\s*(mile|miles|mi)\b", q)
    if m:
        return float(m.group(1))
    m_km = re.search(r"within\s+(\d+(\.\d+)?)\s*(km|kilometers?)\b", q)
    if m_km:
        return float(m_km.group(1)) * 0.621371
    return default_radius_miles


def maybe_route_to_mock_tools(query: str, user_lat: float, user_lon: float, default_radius_miles: float) -> Optional[str]:
    """
    Handles the curated CEO demo prompts + a few general ones.
    Returns a serialized JSON 'tool results' string or None.
    """
    q = query.lower()
    radius = extract_radius_miles(q, default_radius_miles)

    # 1) Indian food near me
    if "indian" in q and ("food" in q or "restaurant" in q or "restaurants" in q):
        results = mock_search_places(
            query="indian restaurant",
            near_lat=user_lat, near_lon=user_lon, radius_miles=radius,
            filters={"cuisine": "Indian"}
        )
        return json.dumps(
            {"tool": "search_places", "args": {"query": "indian restaurant", "radius_miles": radius, "filters": {"cuisine": "Indian"}}, "results": results},
            ensure_ascii=False, indent=2
        )

    # 2) Coach bag -> mall with Coach
    if "coach" in q and ("mall" in q or "store" in q or "shop" in q):
        results = mock_find_brand_in_malls("Coach", user_lat, user_lon, radius)
        return json.dumps(
            {"tool": "find_brand_in_malls", "args": {"brand": "Coach", "radius_miles": radius}, "results": results},
            ensure_ascii=False, indent=2
        )

    # 3) Rooftop cafés open after midnight
    if "rooftop" in q and ("cafe" in q or "cafés" in q or "cafes" in q):
        late = ("after midnight" in q) or ("open after midnight" in q) or ("late" in q)
        results = mock_search_rooftop_cafes(open_after_midnight=late, near_lat=user_lat, near_lon=user_lon, radius_miles=radius)
        return json.dumps(
            {"tool": "search_rooftop_cafes", "args": {"open_after_midnight": late, "radius_miles": radius}, "results": results},
            ensure_ascii=False, indent=2
        )

    # 4) Nearest mall that has both Apple and Zara
    if "mall" in q and ("apple" in q and "zara" in q):
        results = mock_find_malls_with_brands(["Apple", "Zara"], user_lat, user_lon, radius)
        return json.dumps(
            {"tool": "find_malls_with_brands", "args": {"brands": ["Apple", "Zara"], "radius_miles": radius}, "results": results},
            ensure_ascii=False, indent=2
        )

    # 5) Petrol station with a car wash on the way to Diriyah
    if ("petrol" in q or "gas" in q) and "car wash" in q and "diriyah" in q:
        results = mock_petrol_with_carwash_on_way_to_diriyah(user_lat, user_lon, radius)
        return json.dumps(
            {"tool": "find_petrol_with_carwash_diriyah", "args": {"radius_miles": radius}, "results": results},
            ensure_ascii=False, indent=2
        )

    # 6) IMAX near Olaya
    if "imax" in q and "olaya" in q:
        results = mock_find_imax_near(OLAYA[0], OLAYA[1], radius_miles=radius)
        return json.dumps(
            {"tool": "find_imax", "args": {"near": "Olaya", "radius_miles": radius}, "results": results},
            ensure_ascii=False, indent=2
        )

    # 7) Family-friendly restaurants in Riyadh Park Mall
    if ("family" in q or "family-friendly" in q or "kids" in q) and ("riyadh park" in q or "riyadh park mall" in q):
        results = mock_search_places(
            query="family restaurants in mall",
            near_lat=user_lat, near_lon=user_lon, radius_miles=radius,
            filters={"family_friendly": True, "mall": "Riyadh Park Mall"}
        )
        return json.dumps(
            {"tool": "search_places", "args": {"query": "family restaurants in Riyadh Park Mall", "radius_miles": radius, "filters": {"family_friendly": True, "mall": "Riyadh Park Mall"}}, "results": results},
            ensure_ascii=False, indent=2
        )

    # Bonus) Italian near Olaya
    if "italian" in q and ("olaya" in q or "al-olaya" in q or "al olaya" in q or "near me" in q):
        results = mock_search_places(
            query="italian near olaya",
            near_lat=OLAYA[0], near_lon=OLAYA[1], radius_miles=radius,
            filters={"cuisine": "Italian"}
        )
        return json.dumps(
            {"tool": "search_places", "args": {"query": "italian near olaya", "radius_miles": radius, "filters": {"cuisine": "Italian"}}, "results": results},
            ensure_ascii=False, indent=2
        )

    return None


# ---------------------------
# Streamlit App
# ---------------------------

st.set_page_config(page_title="HudHud Riyadh AI Assistant (GPT-5)", page_icon="🗺️", layout="wide")

st.title("🗺️ HudHud Riyadh AI Assistant (GPT-5)")
st.caption("Streaming enabled. Mock tool hooks included. Replace mocks with HudHud APIs for production.")

with st.sidebar:
    st.subheader("Settings")
    model = st.selectbox("OpenAI model", ["gpt-5", "gpt-5-mini", "gpt-4o"], index=0)
    temperature = st.slider("Temperature (ignored for some GPT-5 models)", 0.0, 1.0, 0.3, 0.05)
    allow_stream = st.checkbox("Stream responses", value=True)

    st.markdown("#### Your location (Riyadh only)")
    lat = st.number_input("Latitude", value=RIYADH_CENTER[0], format="%.6f")
    lon = st.number_input("Longitude", value=RIYADH_CENTER[1], format="%.6f")
    default_radius = st.number_input("Default radius (miles)", value=10.0, min_value=1.0, step=1.0)

    use_mock_tools = st.checkbox("Use mock search tools (demo)", value=True, help="Simulate HudHud APIs for brand & place queries.")

    st.divider()
    st.markdown("**💡 Quick test prompts**")
    test_prompts = [
        "What’s your top recommendation for Indian food near me within 10 miles?",
        "I want to buy a Coach bag. Which mall near me has a Coach store?",
        "Are there any rooftop cafés open after midnight in Riyadh?",
        "Where is the nearest mall that has both Apple and Zara stores?",
        "Show me a petrol station with a car wash on the way to Diriyah.",
        "Is there an IMAX cinema near Olaya?",
        "Can you suggest 3 family-friendly restaurants in Riyadh Park Mall?",
    ]
    for i, tp in enumerate(test_prompts):
        if st.button(tp, key=f"tp_{i}"):
            st.session_state["_queued_prompt"] = tp

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Salam! I can help with places in **Riyadh**. What are you looking for?"},
    ]

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message("assistant" if msg["role"] == "assistant" else "user"):
        st.markdown(msg["content"])

# Check for queued quick prompt
queued = st.session_state.pop("_queued_prompt", None)

# Chat input
user_prompt = st.chat_input("Ask about food, shopping, venues, routes—Riyadh only!")

prompt = queued if queued else user_prompt

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    tool_context = None
    if use_mock_tools:
        tool_context = maybe_route_to_mock_tools(prompt, lat, lon, default_radius)

    client = get_openai_client()
    messages = build_messages(st.session_state.messages[:-1], prompt, tool_context)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        acc = ""
        try:
            for delta in stream_openai_response(
                client,
                model=model,
                messages=messages,
                temperature=temperature,
                want_stream=allow_stream,
            ):
                acc += delta
                placeholder.markdown(acc)
        except openai.OpenAIError as e:
            acc = f"*(API error)* {e}"
            placeholder.markdown(acc)

        # Sanitize any residual tool-stub text
        cleaned = sanitize_model_output(acc)
        if cleaned != acc:
            placeholder.markdown(cleaned)

        st.session_state.messages.append({"role": "assistant", "content": cleaned})

st.markdown("---")
st.caption("**Note:** Tool results and POIs are demo-only placeholders. Replace mock functions with HudHud search/POI APIs for production.")
