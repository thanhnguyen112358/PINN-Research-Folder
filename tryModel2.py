# try_model2.py  – run with:  python try_model2.py
# -------------------------------------------------
import pandas as pd
from pathlib import Path
from importlib import import_module
from types import ModuleType

# -------------------------------------------------
# 1) Import the module that contains Model_2
#    Replace "myapp.pipeline" with the actual module path.
# -------------------------------------------------
pipeline: ModuleType = import_module("TextMiningFunctions")   # <-- EDIT ME

Model_2 = pipeline.Model_2      # keep a ref so we can call it later

# -------------------------------------------------
# 2) Provide light-weight stand-ins for the out-of-scope bits
# -------------------------------------------------
# --- updated stub -----------------------------------------------------
import re

def fake_query_model(*args, **kwargs) -> str:
    """
    Minimal stand-in that accepts ANY signature.
    
    Expected call style in Model_2:
        _query_model(messages=[{...}, ...], model=model_id)
    
    Logic:
        • Concatenate all 'content' fields in the messages list.
        • Count how many *distinct* numeric tokens appear.
        • If ≥2 numeric tokens → return "Yes",
          else                → return "No".
    """
    # ----- pull the text the classifier will see ----------------------
    msgs = kwargs.get("messages", [])
    text = " ".join(msg.get("content", "") for msg in msgs)

    # ----- toy heuristic for the “two-parameter” rule -----------------
    numbers = re.findall(r"\d+(?:\.\d+)?", text)     # matches 850, 1.5, 10, 5, etc.
    decision = "Yes" if len(set(numbers)) >= 2 else "No"

    return decision
# --- updated stub -----------------------------------------------------

def fake_Model_1(df: pd.DataFrame) -> pd.DataFrame:
    """
    Place-holder for your real Model_1.
    Here we just tag the rows so we can see the call happened.
    """
    out = df.copy()
    out["model_1_called"] = True
    return out

# -------------------------------------------------
# 3) Monkey-patch those stand-ins onto the *imported* module
#    (Model_2 picks them up through its global namespace)
# -------------------------------------------------
pipeline._query_model = fake_query_model
pipeline.Model_1      = fake_Model_1

# -------------------------------------------------
# 4) Minimal sample data to drive the function
# -------------------------------------------------
sample = pd.DataFrame(
    {
        "content": [
            # ----------------------------- POSITIVES (Yes) -----------------------------
    "The gasification was performed at 850 °C with a steam/biomass ratio of 1.5, using 200 g Fe₂O₃/Al₂O₃ as oxygen carrier in a 3 kWₜₕ bubbling bed for 30 min.",
    "Pine sawdust (15 g) was fed together with 75 g ilmenite (250 µm) at 900 °C and 1 bar; λ = 0.35 and S/B = 0.8 were maintained throughout the 20-min run.",
    "Cellulose powder (10 g) was gasified at 910 °C in a laboratory dual fluidised bed with 100 g CuO–Al₂O₃ carrier; steam flow was 1.2 kg h⁻¹ and residence time 12 min.",
    "A 50 g batch of lignite mixed with 250 g Fe₂O₃ (OC/Fuel = 5 wt %) was processed at 5 bar and 850 °C for 15 min to generate hydrogen-rich syngas.",
    "Hazelnut shell biomass (8 g) and NiO/Al₂O₃ carrier (40 g, 150 µm) were reacted at 880 °C with 50 % steam in N₂; gas residence time ~22 s per cycle.",
    "Rice husk (12 g) with Mn₂O₃/Al₂O₃ carrier (60 g) was gasified at 925 °C, steam/biomass = 1.0, oxygen/fuel λ = 0.4; char conversion reached 98 %.",
    "An ilmenite-supported fluidised bed (2 kWₜₕ) operated at 870 °C, steam flow 18 g min⁻¹, processed 20 g wood pellets with S/B = 0.9 for 10 min.",
    "Corn stover (25 g) was treated in a quartz batch reactor with 125 g Fe₂O₃/Al₂O₃ (250 µm) at 900 °C; steam partial pressure 0.4 bar; run time 30 min.",
    "A lab-scale CLG unit (1.5 kWₜₕ) used CuO-spinels (200 g) and 10 g palm kernel shell at 930 °C, steam/biomass = 0.7; H₂ yield peaked at 53 vol %.",
    "Miscanthus (18 g) gasified for 25 min at 860 °C under 40 % steam with a fresh batch of 70 g Ca-doped Fe₂O₃ carrier (dₚ=180 µm).",
    "Bituminous coal (30 g) mixed with 6 wt % CaCO₃ additive and 150 g NiO/MgO carrier was processed at 900 °C; pressure 3 bar; gas residence 40 s.",
    "Switchgrass (14 g) and 80 g FeTiO₃ carrier cycled 15 times at 895 °C; λ = 0.3; steam feed 0.015 kg min⁻¹; complete burnout after 12 min.",
    "A conical spouted bed (5 cm i.d.) operated at 875 °C, 1 bar, gasified 11 g sawdust with 55 g CuO/ZrO₂ carrier; S/B ratio 1.1; product H₂ 49 vol %.",
    "Red mud-derived OC (90 g) reacted with polypropylene waste (10 g) at 915 °C; steam to carbon ratio 2.0; overall cycle time 18 min.",
    "Fe₂O₃-Co₃O₄ composite carrier (120 g) contacted 22 g sugarcane bagasse at 850 °C and 0.5 MPa; steam 0.02 kg min⁻¹; H₂ selectivity 91 %.",

    # ----------------------------- NEGATIVES (No) ------------------------------
    "Chemical looping gasification has emerged as a promising low-carbon route for producing hydrogen while capturing CO₂ in situ.",
    "Several reviews have discussed the advantages of oxygen carriers compared with traditional gasification catalysts.",
    "Future work will focus on scaling up chemical looping reactors and optimising heat integration.",
    "The thermodynamic feasibility of CLG at high pressure has been widely reported in the literature.",
    "Table 2 compares cold-gas efficiencies reported by various authors using different carrier formulations.",
        ]
    }
)

# -------------------------------------------------
# 5) Run it and show the output
# -------------------------------------------------
if __name__ == "__main__":
    result = Model_2(sample, sleep_s=0)   # sleep_s=0 → faster test
    print("\n=== RESULT ===")
    print(result)