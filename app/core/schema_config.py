"""
app.core.schema_config
───────────────────────
Single source of truth for ALL table configs.

Handcrafted configs for ofbdb and misofb.
Auto-discovered configs loaded from configs/<db_name>.json.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

log = logging.getLogger("app.core.schema_config")


@dataclass
class TableConfig:
    db: str
    table: str
    text_columns: List[str]
    metadata_columns: List[str]
    pk_column: Optional[str]
    label: str
    description: str
    date_column: Optional[str] = None
    file_columns: List[Tuple[str, str]] = field(default_factory=list)


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
        metadata_columns=["id", "user_id", "file_name", "file_type", "file_path", "created_at"],
        date_column="created_at",
        file_columns=[("file_path", "file_type")],
    ),
]

OFBDB_FILE_COLUMNS = {
    "advertisement_items": [("item_file", ""), ("item_specification_file", "")],
    "addendum_items": [("item_file", ""), ("item_specification_file", "")],
    "renewal_items": [("item_file", ""), ("item_specification_file", "")],
    "item_master": [("item_file", ""), ("specification_files", "")],
    "item_master_diff": [("item_file", ""), ("specification_files", "")],
    "vendor_clarification": [("attach_file", "")],
    "vendor_debar": [("debar_file", "")],
}

for _cfg in OFBDB_CONFIGS:
    if _cfg.table in OFBDB_FILE_COLUMNS and not _cfg.file_columns:
        _cfg.file_columns = OFBDB_FILE_COLUMNS[_cfg.table]


MISOFB_CONFIGS: List[TableConfig] = [
    TableConfig(
        db="misofb", table="t_ofbpismas",
        label="OFB Employee Record",
        description="Personnel master: name, grade, trade, PAN, pay, unit (Aadhar excluded)",
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


def get_all_configs() -> List[TableConfig]:
    """
    Merge handcrafted configs with SQLite table_configurations.

    Priority:
      1. Handcrafted configs (OFBDB_CONFIGS, MISOFB_CONFIGS) — always included
      2. SQLite table_configurations (is_selected=1 only) — auto/manual discovered
      3. Legacy JSON configs — fallback for databases not yet in SQLite

    SQLite entries override JSON for the same (db, table) pair.
    Handcrafted configs are filtered by SQLite is_selected if a matching
    row exists.
    """
    from app.core.config_db import get_all_selected_table_configs, is_initialized
    from app.core.schema_intelligence import list_known_databases, load_saved_config

    all_cfgs: List[TableConfig] = list(OFBDB_CONFIGS) + list(MISOFB_CONFIGS)

    seen: set = set()
    for cfg in all_cfgs:
        seen.add((cfg.db, cfg.table))

    if is_initialized():
        sqlite_configs = get_all_selected_table_configs()
        _deselected = _get_deselected_keys()

        filtered_handcrafted = [
            cfg for cfg in all_cfgs
            if (cfg.db, cfg.table) not in _deselected
        ]
        all_cfgs = filtered_handcrafted

        for sc in sqlite_configs:
            key = (sc["db_connection_name"], sc["table_name"])
            if key in seen:
                continue
            all_cfgs.append(TableConfig(
                db=sc["db_connection_name"],
                table=sc["table_name"],
                text_columns=sc.get("text_columns", []),
                metadata_columns=sc.get("metadata_columns", []),
                pk_column=sc.get("pk_column"),
                label=sc.get("label", sc["table_name"]),
                description=sc.get("description", ""),
                date_column=sc.get("date_column"),
                file_columns=[tuple(fc) for fc in sc.get("file_columns", [])],
            ))
            seen.add(key)

    known_dbs = {cfg.db for cfg in all_cfgs}
    for db_name in list_known_databases():
        if db_name in known_dbs:
            continue
        extra = load_saved_config(db_name)
        if extra:
            for cfg in extra:
                key = (cfg.db, cfg.table)
                if key not in seen:
                    all_cfgs.append(cfg)
                    seen.add(key)
            known_dbs.add(db_name)

    return _deduplicate_physical(all_cfgs)


def _deduplicate_physical(cfgs: List[TableConfig]) -> List[TableConfig]:
    """Remove configs that map to the same physical host:port/database + table.

    Handcrafted configs (ofbdb, misofb) come first in the list and always win.
    Unknown logical names pass through unfiltered to avoid data loss.
    """
    from app.core.connection_store import get_connection_store
    from app.config import get_settings

    try:
        store = get_connection_store()
        all_conns = store.load_all()
    except Exception:
        return cfgs

    settings = get_settings()

    physical_map: dict = {}
    for name, cred in all_conns.items():
        physical_map[name] = f"{cred.host}:{cred.port}/{cred.database}"

    if "ofbdb" not in physical_map:
        physical_map["ofbdb"] = (
            f"{settings.ofbdb_host}:{settings.ofbdb_port}/{settings.ofbdb_database}"
        )
    if "misofb" not in physical_map:
        physical_map["misofb"] = (
            f"{settings.misofb_host}:{settings.misofb_port}/{settings.misofb_database}"
        )

    seen_physical: set = set()
    result: list = []
    for cfg in cfgs:
        phys = physical_map.get(cfg.db)
        if phys:
            key = (phys, cfg.table)
            if key in seen_physical:
                log.info(
                    "Skipping duplicate physical table: %s.%s "
                    "(already indexed under another connection name)",
                    cfg.db, cfg.table,
                )
                continue
            seen_physical.add(key)
        result.append(cfg)
    return result


def _get_deselected_keys() -> set:
    """Return set of (db, table) tuples that the user has explicitly deselected."""
    from app.core.config_db import is_initialized
    if not is_initialized():
        return set()
    try:
        from app.core.config_db import _get_conn
        with _get_conn() as conn:
            rows = conn.execute(
                "SELECT db_connection_name, table_name FROM table_configurations WHERE is_selected=0"
            ).fetchall()
            return {(r["db_connection_name"], r["table_name"]) for r in rows}
    except Exception:
        return set()
