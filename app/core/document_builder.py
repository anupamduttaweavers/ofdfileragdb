"""
app.core.document_builder
──────────────────────────
Converts a raw DB row into human-readable text for embedding.

Supports handcrafted templates for known OFB tables and
generic fallback for auto-discovered tables.
"""

import hashlib
import json
from typing import Any, Dict, Optional, Tuple

from app.core.schema_config import TableConfig


def _clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def _join(row: Dict, cols: list, sep: str = " · ") -> str:
    parts = [_clean(row.get(c)) for c in cols]
    return sep.join(p for p in parts if p and p.lower() not in ("none", "null", "nan", "0", ""))


# ─── ofbdb builders ───────────────────────────────────────────────────

def _b_certificate_report(r):
    return (
        f"Vendor Registration Certificate\n"
        f"Vendor: {_clean(r.get('vendor_name'))} | Code: {_clean(r.get('vendor_code'))}\n"
        f"Factory: {_clean(r.get('factory_name'))}\n"
        f"Category: {_clean(r.get('category'))} | Grade: {_clean(r.get('grade'))} | Size: {_clean(r.get('size'))}\n"
        f"Type: {_clean(r.get('reg_type'))} | Application: {_clean(r.get('application_type'))}\n"
        f"Address: {_join(r, ['address','road','city','state'], ', ')}\n"
        f"GST: {_clean(r.get('gst_no'))} | TAN: {_clean(r.get('tan_no'))}\n"
        f"Registered: {_clean(r.get('registration_date'))} | Valid Upto: {_clean(r.get('valid_upto'))}\n"
        f"Items: {_clean(r.get('items'))}"
    )

def _b_certificate_report_product(r):
    return (
        f"Vendor Registered Product\n"
        f"Item: {_clean(r.get('item_name'))} | Code: {_clean(r.get('item_code'))}\n"
        f"Specification: {_clean(r.get('item_specification'))}\n"
        f"Type: {_clean(r.get('item_type'))} | Status: {_clean(r.get('item_status'))}"
    )

def _b_item_master(r):
    tech = _join(r, ["man_tech_1","man_tech_2","man_tech_3",
                     "test_facility_1","test_facility_2","test_facility_3"], " | ")
    return (
        f"Item Master\n"
        f"Item: {_clean(r.get('name'))} | Code: {_clean(r.get('item_code'))}\n"
        f"Drawing No: {_clean(r.get('drawing_no'))} | Unit: {_clean(r.get('item_unit'))} | Type: {_clean(r.get('item_type'))}\n"
        f"Description: {_clean(r.get('item_description'))}\n"
        f"Specification: {_clean(r.get('item_specification'))}\n"
        f"Mfg/Test: {tech}"
    )

def _b_advertisement(r):
    return (
        f"Tender Advertisement\n"
        f"Adv No: {_clean(r.get('adv_number'))}\n"
        f"Quantity: {_clean(r.get('quantity'))} | Approx Value: {_clean(r.get('approx_value'))}\n"
        f"Last Date: {_clean(r.get('last_date'))}\n"
        f"Bank: {_clean(r.get('bank_name'))} | A/C: {_clean(r.get('account_no'))}\n"
        f"Status: {_clean(r.get('adv_status'))} | Published: {_clean(r.get('publish_status'))}\n"
        f"Remark: {_clean(r.get('remark'))}"
    )

def _b_advertisement_items(r):
    tech = _join(r, ["man_tech_1","man_tech_2","test_facility_1","test_facility_2"], " | ")
    return (
        f"Tender Line Item\n"
        f"Item: {_clean(r.get('item_name'))} | Code: {_clean(r.get('item_code'))}\n"
        f"Drawing: {_clean(r.get('drawing_no'))} | Unit: {_clean(r.get('item_unit'))} | Type: {_clean(r.get('item_type'))}\n"
        f"Qty: {_clean(r.get('item_quantity'))} | Approx Value: {_clean(r.get('approx_value'))}\n"
        f"Description: {_clean(r.get('item_description'))}\n"
        f"Spec: {_clean(r.get('item_specification'))}\n"
        f"Mfg/Test: {tech}"
    )

def _b_vendorcom_filtered(r):
    products = _join(r, ["Product","Specification","Product1","Specification1",
                          "Product2","Specification2","Tproduct"], " | ")
    return (
        f"Registered Vendor (Legacy)\n"
        f"Vendor: {_clean(r.get('Name'))} | Code: {_clean(r.get('Vendorcode'))}\n"
        f"Factory: {_clean(r.get('Factory'))} | Officer: {_clean(r.get('Officer'))}\n"
        f"State: {_clean(r.get('State'))} | City: {_clean(r.get('City'))}\n"
        f"Grade: {_clean(r.get('Grade'))} | Category: {_clean(r.get('Categorisation'))} | Size: {_clean(r.get('Size'))}\n"
        f"TAN: {_clean(r.get('VendorTAN'))}\n"
        f"Registered: {_clean(r.get('Dtregistry'))} | Expiry: {_clean(r.get('Dtexpiry'))}\n"
        f"Deemed: {'Yes' if r.get('deemed') else 'No'} | Fin Limit: {_clean(r.get('fin_limit'))}\n"
        f"Products: {products}"
    )

def _b_vendor_registration_certificate(r):
    return (
        f"Vendor Certificate (Portal)\n"
        f"Vendor: {_clean(r.get('vendor_name'))} | Code: {_clean(r.get('vendor_code'))}\n"
        f"Address: {_clean(r.get('vendor_address'))}\n"
        f"Type: {_clean(r.get('reg_type'))} | Cert Type: {_clean(r.get('certificate_type'))}\n"
        f"Registered: {_clean(r.get('registration_date'))} | Valid Upto: {_clean(r.get('valid_upto'))}\n"
        f"Financial Limit: {_clean(r.get('financial_limit'))}\n"
        f"No. Items: {_clean(r.get('no_items'))}\n"
        f"Items: {_clean(r.get('items'))}\n"
        f"Issued By: {_clean(r.get('officer_name'))} ({_clean(r.get('officer_designation'))})"
    )

def _b_vendor_debar(r):
    return (
        f"Debarred Vendor\n"
        f"Vendor ID: {_clean(r.get('vm_id'))} | Report: {_clean(r.get('report_id'))}\n"
        f"Debar Date: {_clean(r.get('debar_date'))}\n"
        f"Reason: {_clean(r.get('reason'))}"
    )

def _b_vendor_master(r):
    return (
        f"Vendor Application\n"
        f"Company: {_clean(r.get('name'))}\n"
        f"Nature: {_clean(r.get('nature_of_company'))} | Industry: {_clean(r.get('category_of_industry'))}\n"
        f"Business: {_clean(r.get('nature_of_business'))}\n"
        f"TAN: {_clean(r.get('tan_no'))} | Excise: {_clean(r.get('excise_registration'))}\n"
        f"Status: {_clean(r.get('orf_status'))} | App Type: {_clean(r.get('application_type'))}\n"
        f"Employees: {_clean(r.get('employee_details'))}\n"
        f"ISO: {_clean(r.get('iso_details'))}"
    )

def _b_factory_master(r):
    return (
        f"OFB Factory\n"
        f"Name: {_clean(r.get('name'))} | Code: {_clean(r.get('factory_code'))} | Abbr: {_clean(r.get('abbreviation'))}\n"
        f"Address: {_join(r, ['address','street','city','state'], ', ')}"
    )

def _b_file_master(r):
    text = (
        f"Uploaded Document\n"
        f"File: {_clean(r.get('file_name'))} | Type: {_clean(r.get('file_type'))}\n"
        f"Uploaded: {_clean(r.get('created_at'))}"
    )
    file_content = _extract_file_content(r)
    if file_content:
        text += f"\n\nFile Content:\n{file_content}"
    return text


def _extract_file_content(r) -> str:
    file_path = _clean(r.get("file_path"))
    file_type = _clean(r.get("file_type"))
    if not file_path:
        return ""
    try:
        from app.services.file_extractor import extract_text_from_file
        content = extract_text_from_file(file_path, file_type=file_type)
        if content:
            max_chars = 5000
            return content[:max_chars] if len(content) > max_chars else content
    except Exception:
        pass
    return ""

# ─── misofb builders ──────────────────────────────────────────────────

def _b_t_ofbpismas(r):
    name = " ".join(filter(None, [_clean(r.get(k)) for k in ["first_name","middle_name","last_name"]]))
    return (
        f"OFB Employee Record\n"
        f"Name: {name} | PNO: {_clean(r.get('pno'))}\n"
        f"Designation: {_clean(r.get('designation'))} | Grade: {_clean(r.get('grade'))} | Trade: {_clean(r.get('trade'))}\n"
        f"Unit: {_clean(r.get('unit'))} | Category: {_clean(r.get('category'))} | Gender: {_clean(r.get('gender'))}\n"
        f"PAN: {_clean(r.get('pan_no'))} | GPF: {_clean(r.get('gpf_no'))} | CPF: {_clean(r.get('cpf_no'))}\n"
        f"Email: {_clean(r.get('email_id'))} | Mobile: {_clean(r.get('mobile_no'))}\n"
        f"Hometown: {_clean(r.get('hometown'))}, {_clean(r.get('home_state'))}\n"
        f"Blood Group: {_clean(r.get('blood_group'))} | Batch Year: {_clean(r.get('batch_year'))}\n"
        f"Pay Level: {_clean(r.get('pay_level'))} | Basic Pay: {_clean(r.get('basic_pay'))}\n"
        f"Date of Joining: {_clean(r.get('dt_joining'))} | DOB: {_clean(r.get('dt_birth'))}"
    )

def _b_consumptionreport(r):
    return (
        f"iGOT Training Record\n"
        f"Employee: {_clean(r.get('Full_Name'))} | Email: {_clean(r.get('Email'))}\n"
        f"Designation: {_clean(r.get('Designation'))} | Ministry: {_clean(r.get('Ministry'))}\n"
        f"Department: {_clean(r.get('Department'))} | Organisation: {_clean(r.get('Organization'))}\n"
        f"Course: {_clean(r.get('Content_Name'))} | Type: {_clean(r.get('Content_Type'))}\n"
        f"Provider: {_clean(r.get('Content_Provider'))}\n"
        f"Batch: {_clean(r.get('Batch_Name'))} ({_clean(r.get('Batch_Start_Date'))} - {_clean(r.get('Batch_End_Date'))})\n"
        f"Status: {_clean(r.get('Status'))} | Progress: {_clean(r.get('Content_Progress_Percentage'))}%\n"
        f"Enrolled: {_clean(r.get('Enrolled_On'))} | Completed: {_clean(r.get('Completed_On'))}\n"
        f"Certificate: {_clean(r.get('Certificate_Generated'))} | Mandatory: {_clean(r.get('Live_CBP_Plan_Mandate'))}\n"
        f"Gender: {_clean(r.get('Gender'))} | Category: {_clean(r.get('Category'))}"
    )

def _b_userreport(r):
    return (
        f"iGOT User Learning Summary\n"
        f"Name: {_clean(r.get('Full_Name'))} | Email: {_clean(r.get('Email'))}\n"
        f"Designation: {_clean(r.get('Designation'))} | Roles: {_clean(r.get('Roles'))}\n"
        f"Ministry: {_clean(r.get('Ministry'))} | Department: {_clean(r.get('Department'))}\n"
        f"Profile: {_clean(r.get('Profile_Status'))}\n"
        f"Course Enrolments: {_clean(r.get('Course_Enrolments'))} | Completions: {_clean(r.get('Course_Completions'))}\n"
        f"Total Learning Hours: {_clean(r.get('Total_Learning_Hours'))}\n"
        f"Karma Points: {_clean(r.get('Karma_Points'))}\n"
        f"Gender: {_clean(r.get('Gender'))} | Category: {_clean(r.get('Category'))}"
    )

def _b_userassessmentreport(r):
    return (
        f"iGOT Assessment Result\n"
        f"Employee: {_clean(r.get('Full_Name'))} | Email: {_clean(r.get('E_mail'))}\n"
        f"Designation: {_clean(r.get('Designation'))} | Ministry: {_clean(r.get('Ministry'))}\n"
        f"Assessment: {_clean(r.get('Assessment_Name'))} | Type: {_clean(r.get('Assessment_Type'))}\n"
        f"Course: {_clean(r.get('Course_Name'))} | Provider: {_clean(r.get('Content_Provider'))}\n"
        f"Score: {_clean(r.get('Latest_Percentage_Achieved'))}% | "
        f"Cut-off: {_clean(r.get('Cut_off_Percentage'))}% | Pass: {_clean(r.get('Pass'))}\n"
        f"Retakes: {_clean(r.get('No_of_Retakes'))} | Last Attempted: {_clean(r.get('Last_Attempted_Date'))}"
    )

def _b_contentreport(r):
    return (
        f"iGOT Course Report\n"
        f"Course: {_clean(r.get('Content_Name'))} | Type: {_clean(r.get('Content_Type'))}\n"
        f"Provider: {_clean(r.get('Content_Provider'))} | Batch: {_clean(r.get('Batch_Name'))}\n"
        f"Status: {_clean(r.get('Content_Status'))}\n"
        f"Enrolled: {_clean(r.get('Enrolled'))} | Not Started: {_clean(r.get('Not_Started'))} | "
        f"In Progress: {_clean(r.get('In_Progress'))} | Completed: {_clean(r.get('Completed'))}\n"
        f"Rating: {_clean(r.get('Content_Rating'))} | Published: {_clean(r.get('Last_Published_On'))}"
    )

def _b_igot_course_master(r):
    return (
        f"iGOT Course Master\n"
        f"Course: {_clean(r.get('course_name'))}\n"
        f"Status: {'Active' if str(r.get('stat')) == '1' else 'Inactive'} | "
        f"Mandatory: {'Yes' if str(r.get('mandatory')) == '1' else 'No'}"
    )

def _b_igo_non_doo_emp(r):
    return (
        f"Non-DOO Employee\n"
        f"Name: {_clean(r.get('Full_Name'))} | Email: {_clean(r.get('Email'))}\n"
        f"Unit: {_clean(r.get('UNIT'))} | Phone: {_clean(r.get('Phone_Number'))}"
    )


_BUILDERS = {
    "certificate_report": _b_certificate_report,
    "certificate_report_product": _b_certificate_report_product,
    "item_master": _b_item_master,
    "advertisement": _b_advertisement,
    "advertisement_items": _b_advertisement_items,
    "vendorcom_filtered": _b_vendorcom_filtered,
    "vendor_registration_certificate": _b_vendor_registration_certificate,
    "vendor_debar": _b_vendor_debar,
    "vendor_master": _b_vendor_master,
    "factory_master": _b_factory_master,
    "file_master": _b_file_master,
    "t_ofbpismas": _b_t_ofbpismas,
    "consumptionreport": _b_consumptionreport,
    "userreport": _b_userreport,
    "userassessmentreport": _b_userassessmentreport,
    "contentreport": _b_contentreport,
    "igot_course_master": _b_igot_course_master,
    "igo_non_doo_emp": _b_igo_non_doo_emp,
}


def build_document(cfg: TableConfig, row: Dict[str, Any]) -> Tuple[str, str, Dict]:
    builder = _BUILDERS.get(cfg.table)
    if builder:
        text = builder(row)
    else:
        parts = []
        for col in cfg.text_columns:
            val = _clean(row.get(col))
            if val and val.lower() not in ("none", "null", "nan", "0", ""):
                label = col.replace("_", " ").title()
                parts.append(f"{label}: {val}")
        text = f"{cfg.label}\n" + "\n".join(parts)

    if cfg.pk_column and row.get(cfg.pk_column) is not None:
        pk_val = str(row[cfg.pk_column])
    else:
        pk_val = hashlib.sha256(text.encode()).hexdigest()[:16]

    doc_id = f"{cfg.db}.{cfg.table}.{pk_val}"

    metadata: Dict[str, Any] = {
        "source_db": cfg.db,
        "source_table": cfg.table,
        "doc_label": cfg.label,
        "pk_value": pk_val,
    }
    for col in cfg.metadata_columns:
        val = row.get(col)
        if val is None:
            continue
        if isinstance(val, (int, float, bool)):
            metadata[col] = val
        elif isinstance(val, (dict, list)):
            metadata[col] = json.dumps(val)
        else:
            metadata[col] = str(val)

    return doc_id, text, metadata
