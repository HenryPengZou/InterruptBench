import base64
import io
import json
import re
import html as _html
from pathlib import Path
from typing import Any

from PIL import Image

from agent.prompts import *
from browser_env import (
    Action,
    ActionTypes,
    ObservationMetadata,
    StateInfo,
    action2str,
)

HTML_TEMPLATE = """
<!DOCTYPE html>
<head>
    <style>
        pre {{
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
    </style>
</head>
<html>
    <body>
     {body}
    </body>
</html>
"""


def get_render_action(
    action: Action,
    observation_metadata: dict[str, ObservationMetadata],
    action_set_tag: str,
) -> str:
    """Parse the predicted actions for rendering purpose. More comprehensive information"""
    match action_set_tag:
        case "id_accessibility_tree":
            text_meta_data = observation_metadata["text"]
            if action["element_id"] in text_meta_data["obs_nodes_info"]:
                node_content = text_meta_data["obs_nodes_info"][
                    action["element_id"]
                ]["text"]
            else:
                node_content = "No match found"

            action_str = f"<div class='raw_parsed_prediction' style='background-color:grey'><pre>{action['raw_prediction']}</pre></div>"
            action_str += f"<div class='action_object' style='background-color:grey'><pre>{repr(action)}</pre></div>"
            action_str += f"<div class='parsed_action' style='background-color:yellow'><pre>{action2str(action, action_set_tag, node_content)}</pre></div>"

        case "som":
            text_meta_data = observation_metadata["text"]
            if action["element_id"] in text_meta_data["obs_nodes_info"]:
                node_content = text_meta_data["obs_nodes_info"][
                    action["element_id"]
                ]
            else:
                node_content = "No match found"
            action_str = f"<div class='raw_parsed_prediction' style='background-color:grey'><pre>{action['raw_prediction']}</pre></div>"
            action_str += f"<div class='action_object' style='background-color:grey'><pre>{repr(action)}</pre></div>"
            action_str += f"<div class='parsed_action' style='background-color:yellow'><pre>{action2str(action, action_set_tag, node_content)}</pre></div>"

        case "playwright":
            action_str = action["pw_code"]
        case "webrl_id":
            action_str = action["raw_prediction"]
        case _:
            raise ValueError(
                f"Unknown action type {action['action_type'], action_set_tag}"
            )
    return action_str


def get_action_description(
    action: Action,
    observation_metadata: dict[str, ObservationMetadata],
    action_set_tag: str,
    prompt_constructor: PromptConstructor | None,
) -> str:
    """Generate the text version of the predicted actions to store in action history for prompt use.
    May contain hint information to recover from the failures"""

    match action_set_tag:
        case "id_accessibility_tree":
            text_meta_data = observation_metadata["text"]
            if action["action_type"] in [
                ActionTypes.CLICK,
                ActionTypes.HOVER,
                ActionTypes.TYPE,
            ]:
                # action_type may be an IntEnum (ActionTypes.CLICK) or a plain int (e.g., from replay JSON).
                try:
                    action_name = ActionTypes(int(action["action_type"])).name.lower()
                except Exception:
                    action_name = str(action["action_type"]).lower()
                if action["element_id"] in text_meta_data["obs_nodes_info"]:
                    node_content = text_meta_data["obs_nodes_info"][
                        action["element_id"]
                    ]["text"]
                    node_content = " ".join(node_content.split()[1:])
                    action_str = action2str(
                        action, action_set_tag, node_content
                    )
                else:
                    action_str = f"Attempt to perfom \"{action_name}\" on element \"[{action['element_id']}]\" but no matching element found. Please check the observation more carefully."
            else:
                if (
                    action["action_type"] == ActionTypes.NONE
                    and prompt_constructor is not None
                ):
                    action_splitter = prompt_constructor.instruction[
                        "meta_data"
                    ]["action_splitter"]
                    action_str = f'The previous prediction you issued was "{action["raw_prediction"]}". However, the format was incorrect. Ensure that the action is wrapped inside a pair of {action_splitter} and enclose arguments within [] as follows: {action_splitter}action [arg] ...{action_splitter}.'
                else:
                    action_str = action2str(action, action_set_tag, "")

        case "som":
            text_meta_data = observation_metadata["image"]
            if action["action_type"] in [
                ActionTypes.CLICK,
                ActionTypes.HOVER,
                ActionTypes.TYPE,
            ]:
                # action_type may be an IntEnum (ActionTypes.CLICK) or a plain int (e.g., from replay JSON).
                try:
                    action_name = ActionTypes(int(action["action_type"])).name.lower()
                except Exception:
                    action_name = str(action["action_type"]).lower()
                if action["element_id"] in text_meta_data["obs_nodes_info"]:
                    action_str = action2str(action, action_set_tag, "")
                else:
                    print(
                        'action["element_id"], text_meta_data["obs_nodes_info"]',
                        action["element_id"],
                        text_meta_data["obs_nodes_info"],
                    )
                    action_str = f"Attempt to perfom \"{action_name}\" on element \"[{action['element_id']}]\" but no matching element found. Please check the observation more carefully."
            else:
                if (
                    action["action_type"] == ActionTypes.NONE
                    and prompt_constructor is not None
                ):
                    action_splitter = prompt_constructor.instruction[
                        "meta_data"
                    ]["action_splitter"]
                    action_str = f'The previous prediction you issued was "{action["raw_prediction"]}". However, the format was incorrect. Ensure that the action is wrapped inside a pair of {action_splitter} and enclose arguments within [] as follows: {action_splitter}action [arg] ...{action_splitter}.'
                else:
                    action_str = action2str(action, action_set_tag, "")

        case "playwright":
            action_str = action["pw_code"]
            
        case "webrl_id":
            action_str = action["raw_prediction"]
            
        case _:
            raise ValueError(f"Unknown action type {action['action_type']}")

    return action_str


class RenderHelper(object):
    """Helper class to render text and image observations and meta data in the trajectory"""

    def __init__(
        self, config_file: str, result_dir: str, action_set_tag: str
    ) -> None:
        with open(config_file, "r") as f:
            _config = json.load(f)
            _config_str = ""
            for k, v in _config.items():
                _config_str += f"{k}: {v}\n"
            _config_str = f"<pre>{_config_str}</pre>\n"
            task_id = _config["task_id"]

        self.action_set_tag = action_set_tag

        self.render_file = open(
            Path(result_dir) / f"render_{task_id}.html", "a+", encoding="utf-8"
        )
        self.render_file.truncate(0)
        # write init template
        self.render_file.write(HTML_TEMPLATE.format(body=f"{_config_str}"))
        self.render_file.read()
        self.render_file.flush()

    def render(
        self,
        action: Action,
        state_info: StateInfo,
        meta_data: dict[str, Any],
        render_screenshot: bool = False,
    ) -> None:
        """Render the trajectory"""
        # text observation
        observation = state_info["observation"]
        text_obs = observation["text"]
        info = state_info["info"]
        new_content = f"<h2>New Page</h2>\n"
        new_content += f"<h3 class='url'><a href={state_info['info']['page'].url}>URL: {state_info['info']['page'].url}</a></h3>\n"
        # IMPORTANT: escape to prevent HTML injection / broken templates
        new_content += (
            f"<div class='state_obv'><pre>{_html.escape(str(text_obs))}</pre><div>\n"
        )

        if render_screenshot:
            # image observation
            img_obs = observation["image"]
            image = Image.fromarray(img_obs)
            byte_io = io.BytesIO()
            image.save(byte_io, format="PNG")
            byte_io.seek(0)
            image_bytes = base64.b64encode(byte_io.read())
            image_str = image_bytes.decode("utf-8")
            new_content += f"<img src='data:image/png;base64,{image_str}' style='width:50vw; height:auto;'/>\n"

        # meta data
        prev_action = meta_data.get("action_history", [""])[-1]
        new_content += (
            "<div class='prev_action' style='background-color:pink'>"
            f"{_html.escape(str(prev_action))}"
            "</div>\n"
        )

        # action
        action_str = get_render_action(
            action,
            info["observation_metadata"],
            action_set_tag=self.action_set_tag,
        )
        # `webrl_id` uses raw LLM outputs; escape to keep HTML well-formed.
        if self.action_set_tag == "webrl_id":
            action_str = f"<pre>{_html.escape(str(action_str))}</pre>"
        # with yellow background
        action_str = f"<div class='predict_action'>{action_str}</div>"
        new_content += f"{action_str}\n"

        # Add new content by inserting before the last </body>.
        # This is more robust than regex because observations may contain "</body>" literally.
        self.render_file.seek(0)
        doc = self.render_file.read()
        close_idx = doc.rfind("</body>")
        if close_idx == -1:
            # Fallback: rewrite whole file using the template.
            doc = HTML_TEMPLATE.format(body=new_content)
        else:
            doc = doc[:close_idx] + new_content + doc[close_idx:]
        self.render_file.seek(0)
        self.render_file.truncate()
        self.render_file.write(doc)
        self.render_file.flush()

    def close(self) -> None:
        self.render_file.close()
