"""Script to generate a synthetic 3-page vehicle owner's manual PDF for testing and demo."""

from pathlib import Path
try:
    import pymupdf as fitz
except ImportError:
    import fitz


def generate_sample_vehicle_manual(output_path: Path) -> Path:
    """Generate a 3-page synthetic car manual with distinct testable topics."""
    doc = fitz.open()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # PAGE 1: Engine Temperature & Overheating
    page1 = doc.new_page(width=595, height=842)  # A4 standard points
    page1.insert_text(fitz.Point(50, 60), "APEX MOTORS — OWNER'S GUIDE", fontsize=10, color=(0.4, 0.4, 0.4))
    page1.insert_text(fitz.Point(50, 100), "Section 1: Engine Cooling System & Temperature Alerts", fontsize=16)
    
    p1_text = (
        "1.1 Engine Temperature Warning Light\n\n"
        "If the engine coolant temperature warning light turns solid RED on the dashboard instrument cluster, "
        "your vehicle is experiencing severe overheating. Continued driving under this condition will cause catastrophic "
        "engine block damage or head gasket failure.\n\n"
        "IMMEDIATE ACTIONS REQUIRED:\n"
        "1. Stop the car safely: Immediately pull over to a safe location away from moving traffic and shift the transmission into Park.\n"
        "2. Turn off the engine: Switch off the ignition immediately to halt heat generation.\n"
        "3. Do not open the hood immediately: Steam or scalding coolant may spray violently under high pressure. Wait at least 15 to 20 minutes for the engine to cool down completely.\n"
        "4. Check coolant level: Once the radiator cap is cool to the touch, inspect the transparent expansion tank reservoir.\n"
        "5. Contact authorized roadside assistance if coolant is leaking beneath the engine cradle or if the warning remains lit upon restart."
    )
    rect1 = fitz.Rect(50, 130, 545, 750)
    page1.insert_textbox(rect1, p1_text, fontsize=11, fontname="helv")
    page1.insert_text(fitz.Point(270, 800), "Page 1", fontsize=9, color=(0.5, 0.5, 0.5))

    # PAGE 2: Tire Maintenance & Pressure Checks
    page2 = doc.new_page(width=595, height=842)
    page2.insert_text(fitz.Point(50, 60), "APEX MOTORS — OWNER'S GUIDE", fontsize=10, color=(0.4, 0.4, 0.4))
    page2.insert_text(fitz.Point(50, 100), "Section 2: Wheels, Tires and Inflation Standards", fontsize=16)

    p2_text = (
        "2.1 Tire Pressure Monitoring System (TPMS) and Maintenance\n\n"
        "Maintaining correct tire pressure ensures optimal braking distances, fuel efficiency, and tire longevity. "
        "Under-inflated tires cause excessive heat buildup and increase the risk of sudden sidewall blowout.\n\n"
        "INSPECTION GUIDELINES:\n"
        "1. Check with cold tires: Always measure tire pressure when tires are cold—meaning the vehicle has been parked for at least 3 hours or driven less than 1 mile (1.6 km).\n"
        "2. Recommended pressure: Refer to the Tire and Loading Information placard located on the driver side B-pillar door jamb. The standard cold tire pressure is 33 psi (228 kPa) for front and rear tires.\n"
        "3. Monthly inspection: Check tire pressures at least once a month and before embarking on extended highway trips.\n"
        "4. Tread depth: Replace tires immediately when tread wear indicators appear across the grooves, indicating less than 2/32 inch (1.6 mm) of remaining tread."
    )
    rect2 = fitz.Rect(50, 130, 545, 750)
    page2.insert_textbox(rect2, p2_text, fontsize=11, fontname="helv")
    page2.insert_text(fitz.Point(270, 800), "Page 2", fontsize=9, color=(0.5, 0.5, 0.5))

    # PAGE 3: 12V Battery & Charging System
    page3 = doc.new_page(width=595, height=842)
    page3.insert_text(fitz.Point(50, 60), "APEX MOTORS — OWNER'S GUIDE", fontsize=10, color=(0.4, 0.4, 0.4))
    page3.insert_text(fitz.Point(50, 100), "Section 3: 12V Electrical Battery and Charging Alerts", fontsize=16)

    p3_text = (
        "3.1 Battery Warning Indicator\n\n"
        "When the red battery icon illuminates continuously while the engine is running, the 12-volt charging system is malfunctioning. "
        "The alternator is no longer recharging the vehicle battery, and the car is operating strictly on residual battery power.\n\n"
        "RECOMMENDED PROCEDURES:\n"
        "1. Conserve electrical power: Immediately turn off all non-essential electrical equipment, including air conditioning, cabin blowers, heated seats, and audio systems.\n"
        "2. Do not stop engine in traffic: If you switch off the engine, there may not be enough electrical charge remaining in the battery to restart the starter motor.\n"
        "3. Drive directly to service: Navigate cautiously to the nearest certified service station before the battery is fully depleted.\n"
        "4. Drive belt inspection: A broken serpentine drive belt often causes sudden charging failure accompanied by loss of power steering assistance."
    )
    rect3 = fitz.Rect(50, 130, 545, 750)
    page3.insert_textbox(rect3, p3_text, fontsize=11, fontname="helv")
    page3.insert_text(fitz.Point(270, 800), "Page 3", fontsize=9, color=(0.5, 0.5, 0.5))

    doc.save(str(output_path))
    doc.close()
    return output_path


if __name__ == "__main__":
    out = Path(__file__).parent / "demo_vehicle_guide.pdf"
    generate_sample_vehicle_manual(out)
    print(f"Sample PDF successfully generated at: {out.resolve()}")
