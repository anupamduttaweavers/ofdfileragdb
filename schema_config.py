"""
schema_config.py
────────────────
Single source of truth for ALL table configs.

How it works:
  • ofbdb and misofb have handcrafted configs (fast, precise, reviewed).
  • Any NEW database uses schema_intelligence.py (Llama 3 8B auto-discovery).
  • All_configs() merges both and is what vectorizer.py uses.

Adding a new DB at runtime:
    python main.py discover --host 192.168.1.10 --user root --db hrms_new
    → LLM inspects schema, generates + saves configs/hrms_new.json
    → Next  python main.py index  picks it up automatically.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TableConfig:
    db:               str
    table:            str
    text_columns:     List[str]
    metadata_columns: List[str]
    pk_column:        Optional[str]
    label:            str
    description:      str
    date_column:      Optional[str] = None


# ══════════════════════════════════════════════════════════════════════
# ofbdb  –  Vendor / OFB procurement portal
# ══════════════════════════════════════════════════════════════════════

OFBDB_CONFIGS: List[TableConfig] = [
    TableConfig(
        db="ofbdb", table="certificate_report",
        label="Vendor Registration Certificate",
        description="Approved vendor certs: grade, category, items, validity",
        pk_column="report_id",
        text_columns=[
            "vendor_name", "name", "vendor_code", "factory_name",
            "address", "road", "city", "state",
            "category", "grade", "size",
            "tan_no", "gst_no", "items",
            "reg_type", "application_type",
        ],
        metadata_columns=[
            "report_id", "factory_id", "factory_name",
            "registration_date", "original_registration_date", "valid_upto",
            "category", "grade", "reg_type",
        ],
        date_column="registration_date",
    ),

    TableConfig(
        db="ofbdb", table="certificate_report_product",
        label="Vendor Registered Product",
        description="Individual products under a vendor certificate",
        pk_column="prod_id",
        text_columns=["item_name", "item_code", "item_specification", "item_type", "item_status"],
        metadata_columns=["prod_id", "report_id", "vm_id", "item_type", "item_status"],
        date_column=None,
    ),

    TableConfig(
        db="ofbdb", table="item_master",
        label="Item Master Catalogue",
        description="Master catalogue with drawing numbers, specs, mfg/test requirements",
        pk_column="id",
        text_columns=[
            "name", "item_code", "item_description", "item_specification",
            "drawing_no", "item_unit", "item_type",
            "man_tech_1", "man_tech_2", "man_tech_3",
            "test_facility_1", "test_facility_2", "test_facility_3",
        ],
        metadata_columns=["id", "factory_id", "item_code", "item_type", "item_status"],
        date_column="created_at",
    ),

    TableConfig(
        db="ofbdb", table="advertisement",
        label="Tender Advertisement",
        description="Published tender adverts with deadline, value, bank details",
        pk_column="id",
        text_columns=["adv_number", "quantity", "remark", "bank_name", "account_no"],
        metadata_columns=[
            "id", "factory_id", "adv_number",
            "last_date", "adv_status", "publish_status", "approx_value",
        ],
        date_column="created_at",
    ),

    TableConfig(
        db="ofbdb", table="advertisement_items",
        label="Advertisement Item (Tender Line)",
        description="Individual items listed in a tender advertisement",
        pk_column="id",
        text_columns=[
            "item_name", "item_code", "drawing_no",
            "item_description", "item_specification",
            "item_unit", "item_type",
            "man_tech_1", "man_tech_2",
            "test_facility_1", "test_facility_2",
        ],
        metadata_columns=["id", "adv_id", "item_code", "item_type", "item_quantity", "approx_value"],
        date_column="created_at",
    ),

    TableConfig(
        db="ofbdb", table="vendorcom_filtered",
        label="Registered Vendor (Legacy)",
        description="Legacy consolidated vendor list with grade, products, expiry dates",
        pk_column="Vendorcode",
        text_columns=[
            "Factory", "Officer", "Name", "Abb",
            "State", "City", "Street",
            "Product", "Specification",
            "Product1", "Specification1",
            "Product2", "Specification2",
            "Tproduct", "VendorTAN",
        ],
        metadata_columns=[
            "Vendorcode", "Factory", "Grade", "Categorisation",
            "Dtregistry", "Dtexpiry", "Size", "deemed",
        ],
        date_column="Dtregistry",
    ),

    TableConfig(
        db="ofbdb", table="vendor_registration_certificate",
        label="Vendor Certificate (Portal)",
        description="Certificates via new portal with financial limits and officer details",
        pk_column="id",
        text_columns=[
            "vendor_name", "vendor_address", "vendor_code",
            "items", "officer_name", "officer_designation",
            "financial_limit", "certificate_type", "reg_type",
        ],
        metadata_columns=[
            "id", "vm_id", "reg_type", "vendor_code",
            "registration_date", "valid_upto",
            "no_items", "certificate_type",
        ],
        date_column="registration_date",
    ),

    TableConfig(
        db="ofbdb", table="vendor_debar",
        label="Debarred Vendor",
        description="Vendors debarred from registration with reason and date",
        pk_column="id",
        text_columns=["reason"],
        metadata_columns=["id", "vm_id", "debar_date", "report_id"],
        date_column="debar_date",
    ),

    TableConfig(
        db="ofbdb", table="vendor_master",
        label="Vendor Application",
        description="Full vendor application with company, financial and technical details",
        pk_column="vm_id",
        text_columns=[
            "name", "nature_of_company", "category_of_industry",
            "nature_of_business", "iso_details", "rd_details",
            "tan_no", "excise_registration",
            "employee_details", "testing_details",
        ],
        metadata_columns=[
            "vm_id", "application_type", "orf_status",
            "tan_no", "factory_reg_no",
        ],
        date_column="created_at",
    ),

    TableConfig(
        db="ofbdb", table="factory_master",
        label="OFB Factory",
        description="Factory name, code, address, state",
        pk_column="id",
        text_columns=["name", "factory_code", "address", "street", "city", "state", "abbreviation"],
        metadata_columns=["id", "factory_code", "factory_status"],
        date_column=None,
    ),

    TableConfig(
        db="ofbdb", table="file_master",
        label="Uploaded Document",
        description="Vendor-uploaded files: PDFs, specs, certificates, drawings",
        pk_column="id",
        text_columns=["file_name", "file_path"],
        metadata_columns=["id", "user_id", "file_name", "file_type", "created_at"],
        date_column="created_at",
    ),
]


# ══════════════════════════════════════════════════════════════════════
# misofb  –  iGOT learning / HR system
# ══════════════════════════════════════════════════════════════════════

MISOFB_CONFIGS: List[TableConfig] = [
    TableConfig(
        db="misofb", table="t_ofbpismas",
        label="OFB Employee Record",
        description="Personnel master: name, grade, trade, PAN, pay, unit  (Aadhar excluded)",
        pk_column="pno",
        text_columns=[
            "first_name", "middle_name", "last_name",
            "designation", "grade", "trade",
            "category", "gender", "email_id",
            "hometown", "home_state",
            "blood_group", "pan_no",
            "batch_year", "gpf_no", "cpf_no", "mode_joining",
        ],
        metadata_columns=[
            "pno", "unit", "grade", "trade", "designation",
            "category", "gender", "pay_level",
            "basic_pay", "gross_pay", "net_pay",
            "dt_birth", "dt_joining",
        ],
        date_column="dt_joining",
    ),

    TableConfig(
        db="misofb", table="consumptionreport",
        label="Training Consumption Record",
        description="Who completed/enrolled in which iGOT course, with certificate info",
        pk_column=None,
        text_columns=[
            "Full_Name", "Designation", "Email",
            "GroupDesig", "Tag", "Ministry", "Department", "Organization",
            "Content_Name", "Content_Type", "Content_Provider",
            "Batch_Name", "Status", "Gender", "Category",
        ],
        metadata_columns=[
            "Email", "Phone_Number",
            "Ministry", "Department", "Organization",
            "Content_Type", "Status",
            "Batch_Id", "Batch_Start_Date", "Batch_End_Date",
            "Enrolled_On", "Completed_On",
            "Content_Progress_Percentage",
            "Certificate_Generated",
            "Live_CBP_Plan_Mandate",
        ],
        date_column="Completed_On",
    ),

    TableConfig(
        db="misofb", table="userreport",
        label="iGOT User Learning Summary",
        description="Per-user enrolments, completions, learning hours, karma points",
        pk_column="Email",
        text_columns=[
            "Full_Name", "Designation", "Email",
            "Group", "Tag", "Ministry", "Department", "Organization",
            "Roles", "Gender", "Category", "Profile_Status",
        ],
        metadata_columns=[
            "Email", "Ministry", "Department", "Organization",
            "Course_Enrolments", "Course_Completions",
            "Total_Learning_Hours", "Karma_Points",
            "User_Registration_Date",
        ],
        date_column="User_Registration_Date",
    ),

    TableConfig(
        db="misofb", table="userassessmentreport",
        label="Assessment Result",
        description="Per-user assessment scores, pass/fail, number of retakes",
        pk_column=None,
        text_columns=[
            "Full_Name", "Designation", "E_mail",
            "Ministry", "Department", "Organisation",
            "Assessment_Name", "Assessment_Type",
            "Content_Provider", "Course_Name",
        ],
        metadata_columns=[
            "E_mail", "Course_ID",
            "Latest_Percentage_Achieved", "Cut_off_Percentage",
            "Pass", "No_of_Retakes",
            "Last_Attempted_Date",
        ],
        date_column="Last_Attempted_Date",
    ),

    TableConfig(
        db="misofb", table="contentreport",
        label="iGOT Course Report",
        description="Course-level stats: enrolled, completed, rating, publish dates",
        pk_column=None,
        text_columns=[
            "Content_Name", "Content_Type", "Content_Provider", "Batch_Name",
        ],
        metadata_columns=[
            "Content_Status", "Content_Duration",
            "Enrolled", "Not_Started", "In_Progress", "Completed",
            "Content_Rating", "Last_Published_On",
        ],
        date_column="Last_Published_On",
    ),

    TableConfig(
        db="misofb", table="igot_course_master",
        label="iGOT Course Master",
        description="Master list of all iGOT courses with mandatory flag",
        pk_column="sl",
        text_columns=["course_name"],
        metadata_columns=["sl", "stat", "mandatory"],
        date_column=None,
    ),

    TableConfig(
        db="misofb", table="igo_non_doo_emp",
        label="Non-DOO Employee",
        description="Employees outside DOO scope tracked for iGOT",
        pk_column="sl",
        text_columns=["Full_Name", "Email", "UNIT"],
        metadata_columns=["sl", "Phone_Number", "UNIT"],
        date_column=None,
    ),
]


# ─────────────────────────────────────────────────────────────────────
# Unified loader — merges hardcoded + all auto-discovered DBs
# ─────────────────────────────────────────────────────────────────────

def get_all_configs() -> List[TableConfig]:
    """
    Returns ALL configs: hardcoded known DBs + any auto-discovered ones
    saved under  configs/<db_name>.json.
    """
    from schema_intelligence import list_known_databases, load_saved_config

    # Start with the two known DBs
    all_cfgs: List[TableConfig] = list(OFBDB_CONFIGS) + list(MISOFB_CONFIGS)
    known_dbs = {"ofbdb", "misofb"}

    # Load any additional auto-discovered DBs
    for db_name in list_known_databases():
        if db_name in known_dbs:
            continue   # Don't double-load the hardcoded ones
        extra = load_saved_config(db_name)
        if extra:
            all_cfgs.extend(extra)
            known_dbs.add(db_name)

    return all_cfgs
