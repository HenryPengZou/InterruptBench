import os
import json
import base64
import time
from .html_parser import HtmlParser
from .configs import basic_attrs
from .scripts import *

class BrowserCrashedError(RuntimeError):
    """Raised when the underlying browser/page target has crashed/closed."""

def _looks_like_target_crash(err: Exception) -> bool:
    msg = str(err)
    return (
        "Target crashed" in msg
        or "has been closed" in msg
        or "Browser has been closed" in msg
        or "closed" == msg.strip().lower()
    )

def _should_save_debug_info() -> bool:
    # Default to off to avoid heavy IO / races during parallel eval.
    return os.environ.get("WEBAGENT_SAVE_DEBUG_INFO", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }

def safe_evaluate(page, expression, arg=None, *, retries: int = 2, base_wait_ms: int = 100):
    """
    Best-effort wrapper around page.evaluate().
    - Retries on transient navigation/context errors.
    - Raises BrowserCrashedError on 'Target crashed'/'closed' errors.
    """
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            if arg is None:
                return page.evaluate(expression)
            return page.evaluate(expression, arg)
        except Exception as e:
            last_err = e
            if _looks_like_target_crash(e):
                raise BrowserCrashedError(str(e)) from e
            # Common transient causes: navigation, execution context destroyed.
            try:
                page.wait_for_load_state("domcontentloaded", timeout=2000)
            except Exception:
                pass
            page.wait_for_timeout(base_wait_ms * (attempt + 1))
    assert last_err is not None
    raise last_err

def safe_screenshot(page, *, path: str | None = None, retries: int = 1, base_wait_ms: int = 100) -> bytes:
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            if path is None:
                return page.screenshot()
            return page.screenshot(path=path)
        except Exception as e:
            last_err = e
            if _looks_like_target_crash(e):
                raise BrowserCrashedError(str(e)) from e
            try:
                page.wait_for_load_state("domcontentloaded", timeout=2000)
            except Exception:
                pass
            page.wait_for_timeout(base_wait_ms * (attempt + 1))
    assert last_err is not None
    raise last_err

def get_window(page):
    x = safe_evaluate(page, "window.scrollX")
    y = safe_evaluate(page, "window.scrollY")
    w = safe_evaluate(page, "window.innerWidth")
    h = safe_evaluate(page, "window.innerHeight")
    return (x, y, w, h)

def modify_page(page):
    page.wait_for_timeout(500)
    try:
        page.wait_for_load_state("domcontentloaded", timeout=5000)
    except Exception:
        pass
    
    try:
        safe_evaluate(page, remove_id_script)
    except Exception:
        pass
    
    packet = {
        "raw_html": safe_evaluate(page, "document.documentElement.outerHTML"),
        "window": get_window(page)
    }
    
    safe_evaluate(page, prepare_script)
    page.wait_for_timeout(100)
    
    # Screenshots are expensive and can crash Chromium under parallel load.
    img_bytes = safe_screenshot(
        page,
        path="debug_info/screenshot_raw.png" if _should_save_debug_info() else None,
    )
    raw_image = base64.b64encode(img_bytes).decode()
    
    safe_evaluate(page, clickable_checker_script)
    page.wait_for_timeout(50)
    
    # get all clickable elements
    start_id = 0
    items, start_id = safe_evaluate(page, label_script, {
        "selector": ".possible-clickable-element",
        "startIndex": start_id
    })
    page.wait_for_timeout(50)
    
    # mark our own labels and get the images
    items = safe_evaluate(page, label_marker_script, items)
    page.wait_for_timeout(100)
    img_bytes = safe_screenshot(
        page,
        path="debug_info/marked.png" if _should_save_debug_info() else None,
    )
    marked_image = base64.b64encode(img_bytes).decode()
    
    # remove markers on the page
    safe_evaluate(page, remove_label_mark_script)
    
    packet.update({
        "raw_image": raw_image,
        "marked_image": marked_image,
        "modified_html": safe_evaluate(page, "document.documentElement.outerHTML")
    })
    
    # element_info, include "all_elements" and "clickable_elements"
    element_info = safe_evaluate(page, element_info_script)
    page.wait_for_timeout(100)
    packet.update(element_info)
    return packet

def save_debug_info(packet):
    if not _should_save_debug_info():
        return
    with open("debug_info/raw.html", "w") as f:
        f.write(packet["modified_html"])
    with open("debug_info/parsed.html", "w") as f:
        f.write(packet["html"])
    with open("debug_info/all_element.json", "w") as f:
        f.write(json.dumps(packet["all_elements"]))
        
def get_parsed_html(page):
    if _should_save_debug_info() and not os.path.exists("debug_info"):
        os.makedirs("debug_info")
        
    print("parsing html...")
    
    packet = modify_page(page)
    raw_html = packet["modified_html"]
    
    args = {
        "use_position": True,
        "rect_dict": {},
        "window_size": packet["window"],
        "id-attr": "data-backend-node-id",
        "label_attr": "data-label-id",
        "label_generator": "order",
        "regenerate_label": False,
        "attr_list": basic_attrs,
        "prompt": "xml",
        "dataset": "pipeline"
    }
    
    hp = HtmlParser(raw_html, args)
    res = hp.parse_tree()
    page_html = res.get("html", "")
    
    packet["html"] = page_html
    
    # for debug
    save_debug_info(packet)
    
    print("parsing finished.")
    
    return packet

