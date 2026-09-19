# scripts/generate_sample_claims.py
"""
Generates rich, multi-page synthetic insurance claim PDFs for the RAG demo.
Uses ReportLab Platypus (multi-page flowables) so each claim becomes a
realistic 3-5 page document with structured sections.
"""

import os
from datetime import date, timedelta
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, KeepTogether,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "claims")

# ---------------------------------------------------------------------------
# RICH SAMPLE DATA
# ---------------------------------------------------------------------------

SAMPLE_CLAIMS = [
    {
        "claim_id": "CLM-1001",
        "policy_number": "POL-88213",
        "claim_date": date(2025, 3, 12),
        "incident_date": date(2025, 3, 10),
        "incident_location": "5th Avenue & Main Street, Springfield, IL",
        "claimant_name": "Jordan A. Miles",
        "claimant_contact": {
            "phone": "(555) 214-8871",
            "email": "jordan.miles@example.com",
            "address": "482 Maple Court, Springfield, IL 62704",
        },
        "claim_type": "Auto Collision",
        "policy_type": "Personal Auto — Full Coverage",
        "policy_start": date(2023, 6, 1),
        "amount_claimed": 4250.00,
        "amount_approved": None,
        "status": "Under Review",
        "description": [
            "The claimant reports being rear-ended while stopped at a traffic "
            "light on the corner of 5th Avenue and Main Street on the morning of "
            "March 10, 2025, at approximately 8:42 AM. Weather conditions were "
            "clear and dry. The claimant states they were stationary for several "
            "seconds when the vehicle behind them failed to brake in time.",
            "The impact pushed the claimant's vehicle forward roughly four feet. "
            "The claimant's 2021 Honda Accord sustained rear bumper deformation, "
            "trunk lid misalignment, and damage to the left rear tail light "
            "assembly. The at-fault driver's 2018 Ford F-150 showed minor front "
            "bumper scuffing.",
            "No injuries were reported at the scene. The claimant declined "
            "ambulance transport but later reported mild neck stiffness on the "
            "evening of March 10. No medical treatment has been sought as of the "
            "claim filing date.",
            "A police report was filed at the scene by Officer R. Delgado "
            "(Badge #4412), Springfield PD. Report number PR-44210 was provided "
            "with the claim submission along with photographs of both vehicles "
            "and a repair estimate from Millbrook Auto Body.",
        ],
        "itemized_losses": [
            ("Rear bumper replacement", 1850.00),
            ("Trunk lid repair and repaint", 950.00),
            ("Left rear tail light assembly", 420.00),
            ("Paint blending and labor", 780.00),
            ("Rental car reimbursement (7 days)", 250.00),
        ],
        "supporting_documents": [
            "Police report PR-44210 (Springfield PD)",
            "Photographs of both vehicles at scene (12 images)",
            "Repair estimate from Millbrook Auto Body, dated 2025-03-11",
            "Copy of at-fault driver's insurance card",
            "Rental car invoice from Enterprise, 7 days",
        ],
        "prior_claims": [
            ("CLM-7734", date(2022, 8, 14), "Auto Collision", 2100.00, "Closed"),
            ("CLM-6120", date(2020, 11, 3), "Windshield Replacement", 480.00, "Closed"),
        ],
        "adjuster_notes": [
            "Initial review completed 2025-03-13 by adjuster M. Okafor. Police "
            "report confirms the narrative provided by the claimant. Photographs "
            "are consistent with a low-speed rear-end impact.",
            "The at-fault driver's insurer has accepted liability. Subrogation "
            "is expected to recover the full claimed amount.",
            "Repair estimate from Millbrook Auto Body is within 6% of market "
            "rates for this vehicle and damage profile. No supplemental estimate "
            "has been requested.",
            "Claimant has been cooperative and responsive to all document "
            "requests. No red flags identified at this stage.",
        ],
        "fraud_indicators": [],
        "communication_log": [
            (date(2025, 3, 12), "Phone", "Initial claim filed by claimant."),
            (date(2025, 3, 13), "Email", "Adjuster requested repair estimate and photos."),
            (date(2025, 3, 14), "Email", "Claimant submitted all requested documents."),
            (date(2025, 3, 15), "Phone", "Adjuster confirmed receipt and initiated subrogation."),
        ],
    },
    {
        "claim_id": "CLM-1002",
        "policy_number": "POL-77410",
        "claim_date": date(2025, 4, 2),
        "incident_date": date(2025, 3, 30),
        "incident_location": "1420 Birch Lane, Naperville, IL",
        "claimant_name": "Priya K. Chandran",
        "claimant_contact": {
            "phone": "(555) 903-2214",
            "email": "priya.chandran@example.com",
            "address": "1420 Birch Lane, Naperville, IL 60540",
        },
        "claim_type": "Home Water Damage",
        "policy_type": "Homeowners — HO-3",
        "policy_start": date(2019, 9, 15),
        "amount_claimed": 9800.00,
        "amount_approved": 9800.00,
        "status": "Approved",
        "description": [
            "The claimant returned from a weekend trip on the evening of "
            "March 30, 2025, and discovered standing water in the kitchen and "
            "adjacent dining area. The source was identified as a burst copper "
            "supply line beneath the kitchen sink, which had failed due to "
            "corrosion and cold-weather stress.",
            "Water had been escaping for an estimated 24 to 36 hours based on "
            "the extent of saturation. The hardwood flooring in the kitchen and "
            "dining area was warped across approximately 220 square feet. The "
            "lower cabinets, toe kicks, and a section of drywall behind the sink "
            "showed significant water damage.",
            "A licensed plumber (Cardinal Plumbing, License #PL-44821) "
            "responded on March 31 and repaired the failed line. An emergency "
            "water extraction crew arrived the same day to remove standing water "
            "and set drying equipment.",
            "No mold growth was observed at the time of inspection. The claimant "
            "has no prior water-damage claims on this policy.",
        ],
        "itemized_losses": [
            ("Hardwood flooring replacement (220 sq ft)", 4400.00),
            ("Lower cabinet replacement (5 units)", 2800.00),
            ("Drywall repair and repaint", 900.00),
            ("Emergency water extraction", 700.00),
            ("Plumbing repair (Cardinal Plumbing)", 450.00),
            ("Dehumidifier rental (5 days)", 350.00),
            ("Temporary storage of displaced items", 200.00),
        ],
        "supporting_documents": [
            "Plumber invoice from Cardinal Plumbing, dated 2025-03-31",
            "Water extraction invoice from ServiceMaster, dated 2025-03-31",
            "Photographs of damage before and after extraction (23 images)",
            "Flooring replacement quote from Hearth & Oak Interiors",
            "Cabinet replacement quote from Kitchen Craft Co.",
        ],
        "prior_claims": [],
        "adjuster_notes": [
            "Field inspection conducted 2025-04-03 by adjuster L. Nguyen. "
            "Observed water staining on subfloor and lower cabinet panels, "
            "consistent with the reported timeline.",
            "Plumber's report confirms the failed section was original copper "
            "piping installed in 1998. No maintenance negligence identified.",
            "All three vendor quotes are within acceptable ranges. Claim "
            "approved for the full amount of $9,800.00.",
            "Payment to be issued to claimant within 5 business days.",
        ],
        "fraud_indicators": [],
        "communication_log": [
            (date(2025, 4, 2), "Phone", "Claim filed by claimant."),
            (date(2025, 4, 3), "In person", "Adjuster field inspection conducted."),
            (date(2025, 4, 4), "Email", "Vendor quotes and photos received."),
            (date(2025, 4, 6), "Email", "Claim approved; payment scheduled."),
        ],
    },
    {
        "claim_id": "CLM-1003",
        "policy_number": "POL-90021",
        "claim_date": date(2025, 1, 20),
        "incident_date": date(2025, 1, 5),
        "incident_location": "Westfield Shopping Center, Oak Brook, IL",
        "claimant_name": "Marcus T. Reyes",
        "claimant_contact": {
            "phone": "(555) 771-9920",
            "email": "marcus.reyes@example.com",
            "address": "88 Sycamore Street, Oak Brook, IL 60523",
        },
        "claim_type": "Auto Theft",
        "policy_type": "Personal Auto — Comprehensive",
        "policy_start": date(2022, 2, 10),
        "amount_claimed": 12500.00,
        "amount_approved": None,
        "status": "Under Review",
        "description": [
            "The claimant's 2019 Toyota Camry was reported stolen from the "
            "Westfield Shopping Center parking lot on the afternoon of "
            "January 5, 2025. The claimant states the vehicle was locked and "
            "parked in the northeast section of the lot at approximately "
            "1:15 PM. The theft was discovered when the claimant returned at "
            "3:45 PM.",
            "The vehicle was recovered by Oak Brook Police on January 14, 2025, "
            "approximately nine days after the theft, in a residential area "
            "roughly six miles from the shopping center. The vehicle showed "
            "significant interior damage, including a torn driver's seat, "
            "damaged dashboard trim, and missing stereo head unit.",
            "The claimant has provided a police report and photographs of the "
            "recovered vehicle. However, the police report number referenced in "
            "the initial claim submission was not included in the attachment, "
            "and this discrepancy is currently being reviewed.",
            "The claimant reported the theft to their insurer on January 6, "
            "2025, one day after the incident.",
        ],
        "itemized_losses": [
            ("Interior upholstery replacement (driver's seat)", 1400.00),
            ("Dashboard trim repair", 850.00),
            ("Stereo head unit replacement", 1200.00),
            ("Steering column repair", 950.00),
            ("Total loss settlement (vehicle value minus salvage)", 8100.00),
        ],
        "supporting_documents": [
            "Photographs of recovered vehicle (18 images)",
            "Oak Brook Police recovery report (pending number confirmation)",
            "Vehicle title (copy)",
            "Loan payoff statement from Toyota Financial Services",
        ],
        "prior_claims": [
            ("CLM-8811", date(2023, 5, 22), "Auto Vandalism", 1800.00, "Closed"),
        ],
        "adjuster_notes": [
            "Review initiated 2025-01-21 by adjuster T. Wallace. Contacted "
            "claimant to request the missing police report number.",
            "Claimant was unable to locate the original police report and has "
            "requested a replacement from Oak Brook PD. This is expected to "
            "arrive within 5–7 business days.",
            "Recovered vehicle inspection scheduled for 2025-01-24 at the "
            "impound lot to verify damage and confirm VIN.",
            "Note: prior vandalism claim in 2023 was closed without issue. "
            "No adverse pattern identified to date. Claim remains under review "
            "pending police documentation.",
        ],
        "fraud_indicators": [
            "Police report number missing from submission",
            "Recovery occurred after 9 days — longer than typical for this area",
            "Vehicle recovered with significant interior damage but no exterior damage",
        ],
        "communication_log": [
            (date(2025, 1, 20), "Phone", "Claim filed by claimant."),
            (date(2025, 1, 21), "Phone", "Adjuster requested missing police report."),
            (date(2025, 1, 22), "Email", "Claimant acknowledged and requested replacement report."),
            (date(2025, 1, 24), "In person", "Vehicle inspection at impound lot."),
        ],
    },
    {
        "claim_id": "CLM-1004",
        "policy_number": "POL-66894",
        "claim_date": date(2025, 5, 15),
        "incident_date": date(2025, 5, 12),
        "incident_location": "302 Elm Street, Evanston, IL",
        "claimant_name": "Elena Fischer",
        "claimant_contact": {
            "phone": "(555) 664-1188",
            "email": "elena.fischer@example.com",
            "address": "302 Elm Street, Evanston, IL 60201",
        },
        "claim_type": "Home Fire Damage",
        "policy_type": "Homeowners — HO-3",
        "policy_start": date(2017, 11, 4),
        "amount_claimed": 21300.00,
        "amount_approved": None,
        "status": "Under Review",
        "description": [
            "On the evening of May 12, 2025, a small kitchen fire broke out "
            "while the claimant was cooking on the stovetop. The claimant "
            "briefly stepped away from the kitchen and returned to find a pan "
            "of oil ignited and flames spreading to nearby cabinetry.",
            "The Evanston Fire Department responded within six minutes of the "
            "9-1-1 call and extinguished the fire. Report #FD-3391 was issued "
            "at the scene. No injuries were reported. The claimant and two "
            "family members were evacuated safely.",
            "Damage was concentrated in the kitchen but smoke spread through "
            "the first-floor living and dining areas. Structural damage to "
            "cabinetry, countertop, and the range hood was extensive. Smoke "
            "residue was observed on walls, ceilings, and furniture throughout "
            "the first floor.",
            "The claimant is currently staying with family nearby while the "
            "property is assessed and remediated.",
        ],
        "itemized_losses": [
            ("Kitchen cabinetry replacement (7 units)", 6800.00),
            ("Countertop replacement (quartz)", 2400.00),
            ("Range hood replacement and installation", 900.00),
            ("Structural framing repair", 3200.00),
            ("Smoke remediation (first floor)", 4500.00),
            ("Interior repainting (kitchen, dining, living)", 2100.00),
            ("Temporary lodging (14 days)", 1400.00),
        ],
        "supporting_documents": [
            "Evanston Fire Department report #FD-3391",
            "Photographs of fire damage (34 images)",
            "Contractor estimate from North Shore Restoration",
            "Lodging receipts (Extended Stay, Evanston)",
            "Inventory of damaged personal property",
        ],
        "prior_claims": [
            ("CLM-4902", date(2021, 3, 9), "Storm Damage", 3400.00, "Closed"),
        ],
        "adjuster_notes": [
            "Field inspection conducted 2025-05-16 by adjuster K. Patel. "
            "Fire department report confirms the origin was unattended cooking.",
            "Structural engineer confirmed the load-bearing elements of the "
            "home were not compromised. Framing repair is localized to the "
            "kitchen ceiling joist area.",
            "Contractor estimate from North Shore Restoration is within 4% of "
            "market rates. Two additional quotes obtained for comparison, both "
            "within 7% of the chosen estimate.",
            "Smoke remediation scope appears appropriate for the observed "
            "damage. Claim remains under review pending final personal property "
            "inventory.",
        ],
        "fraud_indicators": [],
        "communication_log": [
            (date(2025, 5, 15), "Phone", "Claim filed by claimant."),
            (date(2025, 5, 16), "In person", "Adjuster field inspection."),
            (date(2025, 5, 17), "Email", "Contractor estimate and photos received."),
            (date(2025, 5, 19), "Phone", "Discussed personal property inventory."),
        ],
    },
    {
        "claim_id": "CLM-1005",
        "policy_number": "POL-55023",
        "claim_date": date(2025, 6, 8),
        "incident_date": date(2025, 6, 5),
        "incident_location": "Interstate 88, near Aurora, IL",
        "claimant_name": "David O. Nwosu",
        "claimant_contact": {
            "phone": "(555) 442-7701",
            "email": "david.nwosu@example.com",
            "address": "712 Oak Ridge Drive, Aurora, IL 60505",
        },
        "claim_type": "Auto Collision",
        "policy_type": "Personal Auto — Full Coverage",
        "policy_start": date(2024, 1, 12),
        "amount_claimed": 15750.00,
        "amount_approved": 0.00,
        "status": "Denied",
        "description": [
            "The claimant reports being involved in a multi-vehicle collision "
            "on Interstate 88 near Aurora, Illinois, during a heavy rainstorm "
            "on the evening of June 5, 2025. The claimant states their vehicle "
            "was struck from behind by another vehicle, which then pushed them "
            "into the vehicle ahead.",
            "The claimant's 2020 Nissan Altima sustained front-end and "
            "passenger-side damage. Two other vehicles were involved — a 2017 "
            "Chevrolet Malibu and a 2015 Dodge Ram. Damage to the Malibu was "
            "moderate; the Ram sustained minor front bumper damage.",
            "The claimant reports experiencing minor whiplash and sought "
            "medical evaluation on June 6 at Aurora Medical Center. However, "
            "no itemized medical bill was attached to support the claimed "
            "medical portion of $8,500.",
            "The claimant's stated repair estimate is $7,250. No police report "
            "was filed at the scene because the responding officer advised the "
            "parties to exchange information and file online. The claimant has "
            "not provided the online report reference.",
        ],
        "itemized_losses": [
            ("Front bumper and hood repair", 2200.00),
            ("Passenger side door and fender repair", 3100.00),
            ("Headlight and grill replacement", 1150.00),
            ("Medical expenses (claimed, unsupported)", 8500.00),
            ("Pain and suffering (claimed)", 800.00),
        ],
        "supporting_documents": [
            "Repair estimate from Aurora Auto Body",
            "Photographs of claimant's vehicle (8 images)",
            "Aurora Medical Center discharge summary (no itemized bill)",
        ],
        "prior_claims": [
            ("CLM-9124", date(2024, 8, 30), "Auto Collision", 6200.00, "Closed"),
            ("CLM-9345", date(2025, 2, 18), "Auto Collision", 4100.00, "Closed"),
        ],
        "adjuster_notes": [
            "Review initiated 2025-06-09 by adjuster J. Mori. Claimant has "
            "filed two auto collision claims in the past 12 months, both under "
            "similar circumstances (rear-end impacts during rain).",
            "No police report reference provided. The online-reporting system "
            "for Illinois State Police was checked; no matching incident found "
            "for the stated time and location.",
            "Medical expenses claimed at $8,500 with no itemized bill and no "
            "treatment beyond a single ER visit. ER visit typically costs "
            "$1,200–$2,000 for the described evaluation.",
            "Discrepancies between the stated collision narrative and the "
            "damage pattern (front and side damage inconsistent with a single "
            "rear-end impact).",
            "Claim DENIED on 2025-06-20 due to material inconsistencies and "
            "absence of required documentation. Claimant has been notified.",
        ],
        "fraud_indicators": [
            "No police report reference; no matching record found",
            "Two prior collision claims in the past 12 months",
            "Medical expenses claimed without itemized bills",
            "Damage pattern inconsistent with the described incident",
            "Delay in seeking medical treatment (one day after incident)",
            "Claimant declined to provide additional documentation",
        ],
        "communication_log": [
            (date(2025, 6, 8), "Phone", "Claim filed by claimant."),
            (date(2025, 6, 9), "Email", "Adjuster requested police report and medical bills."),
            (date(2025, 6, 12), "Phone", "Claimant unable to provide police reference."),
            (date(2025, 6, 15), "Email", "Second request for documentation."),
            (date(2025, 6, 20), "Phone", "Claim denied; claimant notified."),
        ],
    },
    {
        "claim_id": "CLM-1006",
        "policy_number": "POL-30044",
        "claim_date": date(2025, 2, 27),
        "incident_date": date(2025, 2, 18),
        "incident_location": "905 Willow Court, Skokie, IL",
        "claimant_name": "Sofia Marin",
        "claimant_contact": {
            "phone": "(555) 318-6604",
            "email": "sofia.marin@example.com",
            "address": "905 Willow Court, Skokie, IL 60077",
        },
        "claim_type": "Home Theft / Burglary",
        "policy_type": "Homeowners — HO-3",
        "policy_start": date(2021, 4, 22),
        "amount_claimed": 18200.00,
        "amount_approved": None,
        "status": "Under Review",
        "description": [
            "The claimant reports a burglary occurred while they were away on "
            "vacation from February 14 to February 18, 2025. On returning home "
            "on the evening of February 18, they discovered the front door "
            "unlocked and several items missing from the master bedroom and "
            "home office.",
            "Items reported stolen include a diamond necklace, an emerald "
            "ring, a MacBook Pro, an iPad, and a Nikon DSLR camera with two "
            "lenses. Total claimed value is $18,200.",
            "The claimant states they called the Skokie Police Department on "
            "February 19. The responding officer (Badge #2271) noted no signs "
            "of forced entry. The claimant was unable to locate a copy of the "
            "police report number at the time of filing.",
            "The claimant has provided photographs of the missing items "
            "(taken before the vacation) and purchase receipts for some items.",
        ],
        "itemized_losses": [
            ("Diamond necklace (purchase 2019)", 6500.00),
            ("Emerald ring (purchase 2021)", 4200.00),
            ("MacBook Pro 14\" (2023)", 2400.00),
            ("iPad Pro (2022)", 1100.00),
            ("Nikon DSLR + 2 lenses", 4000.00),
        ],
        "supporting_documents": [
            "Purchase receipt for MacBook Pro (2023)",
            "Purchase receipt for Nikon DSLR (2022)",
            "Photographs of jewelry (2 images)",
            "Skokie PD report reference (pending)",
        ],
        "prior_claims": [
            ("CLM-5110", date(2022, 7, 15), "Home Theft", 3400.00, "Closed"),
        ],
        "adjuster_notes": [
            "Review initiated 2025-03-01 by adjuster R. Hoffman. Noted that "
            "the claimant has a prior theft claim in 2022, also for stolen "
            "jewelry and electronics.",
            "No signs of forced entry reported. Claimant states the front door "
            "may have been left unlocked, but the exact circumstance is unclear.",
            "No police report number provided despite multiple requests. "
            "Skokie PD was contacted directly — no record of the reported "
            "incident under the claimant's name or address.",
            "Jewelry items claimed without professional appraisal. Purchase "
            "receipts for the necklace and ring were not provided.",
            "Claim remains under review pending receipt of the police report "
            "and additional documentation.",
        ],
        "fraud_indicators": [
            "No signs of forced entry",
            "Police report not found in Skokie PD records",
            "High-value jewelry claimed without appraisals or purchase receipts",
            "Prior theft claim in 2022 under similar circumstances",
            "Delay in reporting the theft to police (one day)",
            "Claimant unable to explain how entry was gained",
        ],
        "communication_log": [
            (date(2025, 2, 27), "Phone", "Claim filed by claimant."),
            (date(2025, 3, 1), "Email", "Adjuster requested police report and receipts."),
            (date(2025, 3, 5), "Phone", "Claimant states police report pending."),
            (date(2025, 3, 10), "Phone", "Skokie PD confirms no matching report."),
            (date(2025, 3, 12), "Email", "Additional documentation requested."),
        ],
    },
    {
        "claim_id": "CLM-1007",
        "policy_number": "POL-40199",
        "claim_date": date(2025, 7, 19),
        "incident_date": date(2025, 7, 16),
        "incident_location": "Claimant's residence — 44 Grove Street, Park Ridge, IL",
        "claimant_name": "Benjamin Ho",
        "claimant_contact": {
            "phone": "(555) 882-3317",
            "email": "benjamin.ho@example.com",
            "address": "44 Grove Street, Park Ridge, IL 60068",
        },
        "claim_type": "Auto Vandalism",
        "policy_type": "Personal Auto — Comprehensive",
        "policy_start": date(2023, 9, 5),
        "amount_claimed": 3100.00,
        "amount_approved": None,
        "status": "Under Review",
        "description": [
            "The claimant reports that their 2022 Subaru Outback was vandalized "
            "overnight on July 16, 2025, while parked on the street outside "
            "their residence. According to the claimant, both front-side "
            "windows were broken and the exterior had been spray-painted with "
            "obscene language.",
            "The claimant discovered the damage at approximately 7:00 AM on "
            "July 17 and reported it to Park Ridge PD. A police report was "
            "filed (reference #PR-88214) and provided with the claim.",
            "The repair estimate was obtained from Ho's Body Shop, located in "
            "nearby Niles, IL. The claimant has acknowledged that Ho's Body "
            "Shop is owned by his brother.",
        ],
        "itemized_losses": [
            ("Window glass replacement (2 windows)", 800.00),
            ("Exterior repaint (affected panels)", 1500.00),
            ("Detail and paint correction", 400.00),
            ("Interior cleaning", 200.00),
            ("Rental car (3 days)", 200.00),
        ],
        "supporting_documents": [
            "Park Ridge PD report #PR-88214",
            "Photographs of damage (14 images)",
            "Repair estimate from Ho's Body Shop, Niles, IL",
        ],
        "prior_claims": [],
        "adjuster_notes": [
            "Review initiated 2025-07-21 by adjuster C. Barnes. Park Ridge PD "
            "report confirms the damage described.",
            "Repair estimate obtained from Ho's Body Shop, which is owned by "
            "the claimant's brother. Conflict of interest identified.",
            "Two independent estimates requested. A&P Auto Body quoted $2,150; "
            "Elite Auto quoted $2,050. Both significantly below the submitted "
            "estimate of $3,100.",
            "Claim remains under review pending claimant's response to the "
            "conflict-of-interest notice and the revised estimate.",
        ],
        "fraud_indicators": [
            "Repair estimate from a business owned by claimant's family member",
            "Submitted estimate is ~45% higher than two independent quotes",
        ],
        "communication_log": [
            (date(2025, 7, 19), "Phone", "Claim filed by claimant."),
            (date(2025, 7, 21), "Email", "Adjuster requested independent estimates."),
            (date(2025, 7, 23), "Email", "Conflict-of-interest notice sent to claimant."),
            (date(2025, 7, 26), "Phone", "Claimant acknowledged and will review."),
        ],
    },
    {
        "claim_id": "CLM-1008",
        "policy_number": "POL-20087",
        "claim_date": date(2025, 8, 3),
        "incident_date": date(2025, 8, 1),
        "incident_location": "1180 Lakeview Drive, Highland Park, IL",
        "claimant_name": "Grace Lindqvist",
        "claimant_contact": {
            "phone": "(555) 209-4477",
            "email": "grace.lindqvist@example.com",
            "address": "1180 Lakeview Drive, Highland Park, IL 60035",
        },
        "claim_type": "Home Storm Damage",
        "policy_type": "Homeowners — HO-3",
        "policy_start": date(2018, 3, 19),
        "amount_claimed": 27600.00,
        "amount_approved": None,
        "status": "Approved",
        "description": [
            "On the evening of August 1, 2025, a severe windstorm passed "
            "through Highland Park with gusts recorded up to 62 mph by a "
            "nearby weather station. During the storm, a large oak tree branch "
            "broke and fell onto the roof of the claimant's attached garage, "
            "causing a partial roof collapse.",
            "The affected area covers roughly 300 square feet of the garage "
            "roof structure, including rafters, sheathing, and shingles. No "
            "damage was sustained to the main residence. The claimant's vehicle "
            "was not in the garage at the time of the incident.",
            "A contractor from Lakeview Roofing & Restoration inspected the "
            "damage on August 2 and provided a repair estimate. Photographs "
            "were taken before and after debris removal.",
            "No injuries were reported. The claimant remained in the residence "
            "throughout. The damaged garage is currently tarped pending repair.",
        ],
        "itemized_losses": [
            ("Roof structure repair (300 sq ft)", 9800.00),
            ("Rafter and sheathing replacement", 6200.00),
            ("Shingle replacement", 3400.00),
            ("Gutter and downspout repair", 1200.00),
            ("Tree debris removal", 1800.00),
            ("Temporary tarping and stabilization", 900.00),
            ("Structural engineer inspection", 800.00),
            ("Interior garage repair (drywall, paint)", 3500.00),
        ],
        "supporting_documents": [
            "Photographs of damage before and after debris removal (28 images)",
            "Lakeview Roofing & Restoration estimate",
            "Structural engineer report (signed)",
            "Weather station wind data (62 mph gust confirmation)",
            "Tree removal invoice",
        ],
        "prior_claims": [
            ("CLM-3221", date(2019, 6, 12), "Hail Damage", 2400.00, "Closed"),
        ],
        "adjuster_notes": [
            "Field inspection conducted 2025-08-05 by adjuster S. Ahmed. "
            "Damage is consistent with the described wind event. Weather data "
            "confirms 62 mph gusts at the time reported.",
            "Structural engineer confirmed no compromise to the load-bearing "
            "walls of the residence. Damage isolated to the garage roof.",
            "Lakeview Roofing estimate is competitive; two additional quotes "
            "were obtained (both within 6% of Lakeview's number).",
            "Claim APPROVED on 2025-08-08 for the full amount of $27,600.00. "
            "Payment to be issued within 7 business days.",
        ],
        "fraud_indicators": [],
        "communication_log": [
            (date(2025, 8, 3), "Phone", "Claim filed by claimant."),
            (date(2025, 8, 5), "In person", "Adjuster field inspection."),
            (date(2025, 8, 6), "Email", "Contractor estimate and photos received."),
            (date(2025, 8, 8), "Phone", "Claim approved; payment scheduled."),
        ],
    },
]


# ---------------------------------------------------------------------------
# PDF RENDERING
# ---------------------------------------------------------------------------

styles = getSampleStyleSheet()

H1 = ParagraphStyle(
    "H1", parent=styles["Heading1"], fontSize=16, spaceAfter=6, textColor=colors.HexColor("#1a2b4c")
)
H2 = ParagraphStyle(
    "H2", parent=styles["Heading2"], fontSize=12, spaceBefore=10, spaceAfter=4,
    textColor=colors.HexColor("#2a3b6c"),
)
BODY = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=10, leading=14, alignment=TA_LEFT)
SMALL = ParagraphStyle("Small", parent=styles["BodyText"], fontSize=8, textColor=colors.grey)
LABEL = ParagraphStyle("Label", parent=styles["BodyText"], fontSize=10, leading=14, fontName="Helvetica-Bold")
VALUE = ParagraphStyle("Value", parent=styles["BodyText"], fontSize=10, leading=14)


def _kv_table(rows, col_widths=(1.6 * inch, 4.5 * inch)):
    data = [[Paragraph(k, LABEL), Paragraph(str(v), VALUE)] for k, v in rows]
    t = Table(data, colWidths=col_widths, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
    ]))
    return t


def _bullet_list(items):
    return [Paragraph(f"• {item}", BODY) for item in items]


def _money(x):
    return f"${x:,.2f}"


def build_pdf(claim: dict, output_path: str) -> None:
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        title=f"Insurance Claim {claim['claim_id']}",
        author="Synthetic Data Generator",
    )

    story = []

    # ---- Header ----
    story.append(Paragraph("INSURANCE CLAIM RECORD", H1))
    story.append(Paragraph(
        f"Claim ID: <b>{claim['claim_id']}</b> &nbsp;&nbsp; | &nbsp;&nbsp; "
        f"Policy: <b>{claim['policy_number']}</b> &nbsp;&nbsp; | &nbsp;&nbsp; "
        f"Status: <b>{claim['status']}</b>",
        BODY,
    ))
    story.append(Paragraph(
        "SYNTHETIC DATA — FOR TESTING PURPOSES ONLY",
        SMALL,
    ))
    story.append(Spacer(1, 12))

    # ---- Section 1: Claim Summary ----
    story.append(Paragraph("1. Claim Summary", H2))
    story.append(_kv_table([
        ("Claim Type", claim["claim_type"]),
        ("Policy Type", claim["policy_type"]),
        ("Policy Start Date", claim["policy_start"].isoformat()),
        ("Claim Filed", claim["claim_date"].isoformat()),
        ("Incident Date", claim["incident_date"].isoformat()),
        ("Incident Location", claim["incident_location"]),
        ("Amount Claimed", _money(claim["amount_claimed"])),
        ("Amount Approved", _money(claim["amount_approved"]) if claim["amount_approved"] is not None else "Pending"),
        ("Current Status", claim["status"]),
    ]))
    story.append(Spacer(1, 8))

    # ---- Section 2: Claimant Information ----
    story.append(Paragraph("2. Claimant Information", H2))
    story.append(_kv_table([
        ("Name", claim["claimant_name"]),
        ("Phone", claim["claimant_contact"]["phone"]),
        ("Email", claim["claimant_contact"]["email"]),
        ("Address", claim["claimant_contact"]["address"]),
    ]))
    story.append(Spacer(1, 8))

    # ---- Section 3: Incident Description ----
    story.append(Paragraph("3. Incident Description", H2))
    for para in claim["description"]:
        story.append(Paragraph(para, BODY))
        story.append(Spacer(1, 6))

    story.append(PageBreak())

    # ---- Section 4: Itemized Losses ----
    story.append(Paragraph("4. Itemized Losses", H2))
    loss_rows = [[Paragraph("<b>Item</b>", BODY), Paragraph("<b>Amount</b>", BODY)]]
    for item, amt in claim["itemized_losses"]:
        loss_rows.append([Paragraph(item, BODY), Paragraph(_money(amt), BODY)])
    loss_rows.append([Paragraph("<b>Total Claimed</b>", BODY), Paragraph(f"<b>{_money(claim['amount_claimed'])}</b>", BODY)])
    t = Table(loss_rows, colWidths=(4.2 * inch, 1.8 * inch), hAlign="LEFT")
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.grey),
        ("LINEABOVE", (0, -1), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#f0f3fa")),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))

    # ---- Section 5: Supporting Documentation ----
    story.append(Paragraph("5. Supporting Documentation", H2))
    for b in _bullet_list(claim["supporting_documents"]):
        story.append(b)
    story.append(Spacer(1, 10))

    # ---- Section 6: Prior Claims History ----
    story.append(Paragraph("6. Prior Claims History", H2))
    if claim["prior_claims"]:
        prior_rows = [[
            Paragraph("<b>Claim ID</b>", BODY),
            Paragraph("<b>Date</b>", BODY),
            Paragraph("<b>Type</b>", BODY),
            Paragraph("<b>Amount</b>", BODY),
            Paragraph("<b>Status</b>", BODY),
        ]]
        for cid, dt, ct, amt, st in claim["prior_claims"]:
            prior_rows.append([
                Paragraph(cid, BODY),
                Paragraph(dt.isoformat(), BODY),
                Paragraph(ct, BODY),
                Paragraph(_money(amt), BODY),
                Paragraph(st, BODY),
            ])
        pt = Table(prior_rows, colWidths=(1.0 * inch, 1.0 * inch, 1.6 * inch, 1.0 * inch, 0.9 * inch), hAlign="LEFT")
        pt.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.grey),
        ]))
        story.append(pt)
    else:
        story.append(Paragraph("No prior claims on this policy.", BODY))

    story.append(PageBreak())

    # ---- Section 7: Adjuster Notes ----
    story.append(Paragraph("7. Adjuster Notes", H2))
    for note in claim["adjuster_notes"]:
        story.append(Paragraph(note, BODY))
        story.append(Spacer(1, 6))

    # ---- Section 8: Fraud Indicators ----
    story.append(Paragraph("8. Fraud Indicators / Red Flags", H2))
    if claim["fraud_indicators"]:
        for b in _bullet_list(claim["fraud_indicators"]):
            story.append(b)
    else:
        story.append(Paragraph("None identified at this stage.", BODY))
    story.append(Spacer(1, 10))

    # ---- Section 9: Communication Log ----
    story.append(Paragraph("9. Communication Log", H2))
    comm_rows = [[
        Paragraph("<b>Date</b>", BODY),
        Paragraph("<b>Channel</b>", BODY),
        Paragraph("<b>Note</b>", BODY),
    ]]
    for dt, ch, note in claim["communication_log"]:
        comm_rows.append([Paragraph(dt.isoformat(), BODY), Paragraph(ch, BODY), Paragraph(note, BODY)])
    ct_table = Table(comm_rows, colWidths=(1.0 * inch, 1.0 * inch, 4.5 * inch), hAlign="LEFT")
    ct_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.grey),
    ]))
    story.append(ct_table)
    story.append(Spacer(1, 14))

    story.append(Paragraph(
        f"— End of Claim Record {claim['claim_id']} —",
        SMALL,
    ))

    doc.build(story)


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