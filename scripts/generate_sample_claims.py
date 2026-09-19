import os
from datetime import date
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "claims")

SAMPLE_CLAIMS = [
    {
        "claim_id": "CLM-1001",
        "policy_number": "POL-88213",
        "claim_date": date(2025, 3, 12),
        "claimant_name": "Jordan A. Miles",
        "claim_type": "Auto Collision",
        "description": (
            "Claimant reports a rear-end collision at a stoplight on 5th Ave. "
            "Vehicle sustained rear bumper and trunk damage. No injuries reported. "
            "Police report #PR-44210 was filed at the scene. Repair estimate "
            "obtained from Millbrook Auto Body."
        ),
        "amount_claimed": 4250.00,
        "status": "Under Review",
    },
    {
        "claim_id": "CLM-1002",
        "policy_number": "POL-77410",
        "claim_date": date(2025, 4, 2),
        "claimant_name": "Priya K. Chandran",
        "claim_type": "Home Water Damage",
        "description": (
            "A burst pipe under the kitchen sink caused water damage to flooring "
            "and lower cabinets. Claimant discovered the damage on returning from "
            "a weekend trip. Plumber invoice attached. No prior related claims on file."
        ),
        "amount_claimed": 9800.00,
        "status": "Approved",
    },
    {
        "claim_id": "CLM-1003",
        "policy_number": "POL-90021",
        "claim_date": date(2025, 1, 20),
        "claimant_name": "Marcus T. Reyes",
        "claim_type": "Auto Theft",
        "description": (
            "Claimant's vehicle was reported stolen from a shopping center parking "
            "lot. Vehicle recovered 9 days later with significant interior damage "
            "and missing stereo equipment. Claimant provided a police report and "
            "photos of recovered vehicle condition, but the police report number "
            "was not included in the submission."
        ),
        "amount_claimed": 12500.00,
        "status": "Under Review",
    },
    {
        "claim_id": "CLM-1004",
        "policy_number": "POL-66894",
        "claim_date": date(2025, 5, 15),
        "claimant_name": "Elena Fischer",
        "claim_type": "Home Fire Damage",
        "description": (
            "Small kitchen fire caused by unattended stovetop cooking resulted in "
            "smoke damage throughout the first floor and structural damage to "
            "kitchen cabinetry. Fire department responded and extinguished the fire. "
            "Report #FD-3391 attached."
        ),
        "amount_claimed": 21300.00,
        "status": "Under Review",
    },
    {
        "claim_id": "CLM-1005",
        "policy_number": "POL-55023",
        "claim_date": date(2025, 6, 8),
        "claimant_name": "David O. Nwosu",
        "claim_type": "Auto Collision",
        "description": (
            "Multi-vehicle collision on the highway during heavy rain. Claimant's "
            "vehicle sustained front-end and side damage. Two other vehicles were "
            "involved. Claimant reports minor whiplash and sought medical evaluation "
            "the following day, but no itemized medical bill was attached to support "
            "the claimed amount."
        ),
        "amount_claimed": 15750.00,
        "status": "Denied",
    },
    {
        "claim_id": "CLM-1006",
        "policy_number": "POL-30044",
        "claim_date": date(2025, 2, 27),
        "claimant_name": "Sofia Marin",
        "claim_type": "Home Theft/Burglary",
        "description": (
            "Claimant reports a burglary while away on vacation. Items reported "
            "stolen include jewelry and electronics. No forced entry was noted by "
            "the responding officer, and no police report number was provided with "
            "the claim submission."
        ),
        "amount_claimed": 18200.00,
        "status": "Under Review",
    },
    {
        "claim_id": "CLM-1007",
        "policy_number": "POL-40199",
        "claim_date": date(2025, 7, 19),
        "claimant_name": "Benjamin Ho",
        "claim_type": "Auto Vandalism",
        "description": (
            "Claimant's vehicle windows were broken and the exterior was "
            "spray-painted while parked overnight outside claimant's residence. "
            "The repair estimate was submitted by Ho's Body Shop, which is owned "
            "by the claimant's brother."
        ),
        "amount_claimed": 3100.00,
        "status": "Under Review",
    },
    {
        "claim_id": "CLM-1008",
        "policy_number": "POL-20087",
        "claim_date": date(2025, 8, 3),
        "claimant_name": "Grace Lindqvist",
        "claim_type": "Home Storm Damage",
        "description": (
            "Severe windstorm caused a large tree branch to fall on the roof, "
            "resulting in a partial roof collapse over the garage. Contractor "
            "inspection report and photos of the damage were submitted along with "
            "a repair estimate of the affected area."
        ),
        "amount_claimed": 27600.00,
        "status": "Approved",
    },
]


def build_pdf(claim: dict, output_path: str) -> None:
    """Draws a single claim record onto a one-page PDF."""
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, height - 72, "INSURANCE CLAIM RECORD")
    c.setFont("Helvetica", 10)
    c.drawString(72, height - 90, "SYNTHETIC DATA -- FOR TESTING PURPOSES ONLY")

    c.setFont("Helvetica-Bold", 11)
    y = height - 130
    line_height = 20

    fields = [
        ("Claim ID", claim["claim_id"]),
        ("Policy Number", claim["policy_number"]),
        ("Claim Date", claim["claim_date"].isoformat()),
        ("Claimant Name", claim["claimant_name"]),
        ("Claim Type", claim["claim_type"]),
        ("Amount Claimed", f"${claim['amount_claimed']:,.2f}"),
        ("Status", claim["status"]),
    ]

    for label, value in fields:
        c.setFont("Helvetica-Bold", 11)
        c.drawString(72, y, f"{label}:")
        c.setFont("Helvetica", 11)
        c.drawString(220, y, str(value))
        y -= line_height

    y -= 10
    c.setFont("Helvetica-Bold", 11)
    c.drawString(72, y, "Description:")
    y -= line_height

    c.setFont("Helvetica", 10)
    description = claim["description"]
    max_chars_per_line = 95
    words = description.split()
    current_line = ""
    for word in words:
        if len(current_line) + len(word) + 1 > max_chars_per_line:
            c.drawString(72, y, current_line)
            y -= 14
            current_line = word
        else:
            current_line = f"{current_line} {word}".strip()
    if current_line:
        c.drawString(72, y, current_line)

    c.showPage()
    c.save()


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for claim in SAMPLE_CLAIMS:
        filename = f"{claim['claim_id']}.pdf"
        output_path = os.path.join(OUTPUT_DIR, filename)
        build_pdf(claim, output_path)
        print(f"Created: {output_path}")
    print(f"\nDone. {len(SAMPLE_CLAIMS)} synthetic claim PDFs written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
