import os
import unittest
from unittest.mock import patch, MagicMock
from gui_agent.key_extractor import extract_all_keys, extract_gemini_api_key
from gui_agent.display import DualMonitorLock, DisplayGeometry
from gui_agent.actions import OSActionHandler

class TestGUIAgent(unittest.TestCase):

    def test_multi_provider_key_extraction(self):
        keystore = extract_all_keys()
        self.assertGreaterEqual(len(keystore.openrouter_keys), 1)
        self.assertTrue(keystore.get_openrouter_key().startswith("sk-or-v1-"))
        self.assertIsNotNone(keystore.get_nvidia_key())
        self.assertTrue(keystore.get_nvidia_key().startswith("nvapi-"))
        self.assertGreaterEqual(len(keystore.cerebras_keys), 1)
        self.assertGreaterEqual(len(keystore.mistral_keys), 1)
        self.assertGreaterEqual(len(keystore.explorium_keys), 1)
        self.assertGreaterEqual(len(keystore.groq_keys), 1)
        self.assertTrue(keystore.get_groq_key().startswith("gsk_"))

        # Check that Gemini key is not the exhausted ...U604 key
        gem_key = keystore.get_gemini_key()
        self.assertIsNotNone(gem_key)
        self.assertFalse(gem_key.endswith("U604"), "Must not return exhausted ...U604 key")

    def test_dual_monitor_detection_and_locking(self):
        lock = DualMonitorLock()
        self.assertEqual(lock.primary.name, "LVDS-1")
        self.assertEqual(lock.secondary.name, "VGA-1")
        self.assertEqual(lock.secondary.bbox, (1366, 0, 3286, 1080))

    def test_coordinate_translation_and_safety_checks(self):
        lock = DualMonitorLock()
        gx, gy = lock.translate_and_validate(100, 200)
        self.assertEqual(gx, 1366 + 100)
        self.assertEqual(gy, 0 + 200)

        with self.assertRaises(ValueError):
            lock.translate_and_validate(2000, 500)

    def test_wait_tool_execution(self):
        lock = DualMonitorLock()
        handler = OSActionHandler(lock)
        res = handler.wait(0.1)
        self.assertIn("0.1", res)

    def test_groq_primary_model_and_action_parsing(self):
        from gui_agent.agent import GroqProvider
        provider = GroqProvider(api_key="gsk_fake_key_123")
        self.assertEqual(
            provider.DEFAULT_MODELS[0],
            "meta-llama/llama-3.2-11b-vision-preview",
            "Primary Groq model must be meta-llama/llama-3.2-11b-vision-preview",
        )

        lock = DualMonitorLock()
        handler = OSActionHandler(lock)

        # Test parsing JSON action block from text
        text_json = 'I will now press the super key: {"action": "wait", "args": {"seconds": 0.5}}'
        success = provider._parse_text_action(text_json, handler)
        self.assertTrue(success)

        # Test parsing click action syntax
        with patch.object(handler, "click") as mock_click:
            success = provider._parse_text_action("I am calling click(120, 340)", handler)
            self.assertTrue(success)
            mock_click.assert_called_once_with(120, 340)

        # Test parsing press_key syntax
        with patch.object(handler, "press_key") as mock_press:
            success = provider._parse_text_action("press_key('Super')", handler)
            self.assertTrue(success)
            mock_press.assert_called_once_with("Super")

        # Test parsing natural language Super press
        with patch.object(handler, "press_key") as mock_press_nl:
            success = provider._parse_text_action("I am pressing the Super key now", handler)
            self.assertTrue(success)
            mock_press_nl.assert_called_once_with("Super")

    def test_gemini_key_rotation_and_env_isolation(self):
        import os
        from gui_agent.agent import GeminiProvider

        test_keys = ["AIzaSyFakeKeyOne1111111", "AIzaSyFakeKeyTwo2222222", "AIzaSyFakeKeyThree3333333"]
        os.environ["GOOGLE_API_KEY"] = "exhausted_env_key"
        os.environ["GEMINI_API_KEY"] = "exhausted_env_key"

        with patch("google.genai.Client") as mock_client:
            provider = GeminiProvider(api_keys=test_keys)
            self.assertEqual(provider.current_key_idx, 0)
            self.assertEqual(os.environ["GOOGLE_API_KEY"], test_keys[0])
            self.assertEqual(os.environ["GEMINI_API_KEY"], test_keys[0])

            # Rotate once
            rotated = provider._rotate_key()
            self.assertTrue(rotated)
            self.assertEqual(provider.current_key_idx, 1)
            self.assertEqual(os.environ["GOOGLE_API_KEY"], test_keys[1])
            self.assertEqual(os.environ["GEMINI_API_KEY"], test_keys[1])

            # Rotate twice
            rotated = provider._rotate_key()
            self.assertTrue(rotated)
            self.assertEqual(provider.current_key_idx, 2)
            self.assertEqual(os.environ["GOOGLE_API_KEY"], test_keys[2])

            # Further rotation when pool exhausted
            rotated = provider._rotate_key()
            self.assertFalse(rotated)

    def test_bytez_key_extraction_and_provider(self):
        from gui_agent.agent import BytezProvider
        keystore = extract_all_keys()
        self.assertGreaterEqual(len(keystore.bytez_keys), 1)
        self.assertIsNotNone(keystore.get_bytez_key())

        provider = BytezProvider(api_key=keystore.get_bytez_key())
        self.assertEqual(provider.DEFAULT_MODELS[0], "Qwen/Qwen2.5-VL-7B-Instruct")

    def test_prompt_architect_refinement(self):
        from gui_agent.architect import PromptArchitect
        keystore = extract_all_keys()
        architect = PromptArchitect(keystore)
        refined = architect._heuristic_refine("Take test on screen 2")
        self.assertIn("antiX Linux X11", refined)
        self.assertIn("Execution Steps", refined)

    def test_visual_scaling_alignment(self):
        lock = DualMonitorLock()
        # Ensure scale factors are 1.0 when capture matches display size
        lock.last_capture_size = (lock.secondary.width, lock.secondary.height)
        gx, gy = lock.translate_and_validate(100, 100)
        self.assertEqual(gx, lock.secondary.offset_x + 100)
        self.assertEqual(gy, lock.secondary.offset_y + 100)

        # Ensure half-resolution screenshot scales up correctly without drift
        lock.last_capture_size = (lock.secondary.width // 2, lock.secondary.height // 2)
        gx2, gy2 = lock.translate_and_validate(50, 50)
        self.assertEqual(gx2, lock.secondary.offset_x + 100)
        self.assertEqual(gy2, lock.secondary.offset_y + 100)

    def test_ocr_target_snapping_fallback(self):
        lock = DualMonitorLock()
        handler = OSActionHandler(lock)
        # Should gracefully return original coords if image has no text
        sx, sy = handler._snap_coordinates_ocr(100, 100)
        self.assertEqual(sx, 100)
        self.assertEqual(sy, 100)

    def test_config_json_persistence(self):
        import tempfile
        keystore = extract_all_keys()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            keystore.save_to_config(tmp_path)
            new_keystore = extract_all_keys(config_path=tmp_path)
            self.assertEqual(len(new_keystore.groq_keys), len(keystore.groq_keys))
            self.assertEqual(len(new_keystore.bytez_keys), len(keystore.bytez_keys))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_agnes_key_extraction_and_provider_setup(self):
        from gui_agent.agent import AgnesProvider
        keystore = extract_all_keys()
        self.assertGreaterEqual(len(keystore.agnes_keys), 1)
        self.assertIsNotNone(keystore.get_agnes_key())

        provider = AgnesProvider(api_key=keystore.get_agnes_key())
        self.assertEqual(provider.DEFAULT_MODELS[0], "agnes-2.5-flash")
        self.assertEqual(provider.DEFAULT_MODELS[1], "agnes-2.0-flash")
        self.assertEqual(provider.client.base_url, "https://apihub.agnes-ai.com/v1/")

    def test_agnes_single_image_sanitization(self):
        from gui_agent.agent import AgnesProvider
        provider = AgnesProvider(api_key="sk-test-key")
        
        provider.messages = [
            {"role": "user", "content": [
                {"type": "text", "text": "Step 1"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,OLD1"}}
            ]},
            {"role": "assistant", "content": "Clicked"},
            {"role": "user", "content": [
                {"type": "text", "text": "Step 2"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,OLD2"}}
            ]},
        ]
        
        provider._strip_historical_images()
        
        # Historical user messages must have their image_urls stripped, retaining text
        self.assertEqual(len(provider.messages[0]["content"]), 1)
        self.assertEqual(provider.messages[0]["content"][0]["type"], "text")
        self.assertEqual(provider.messages[0]["content"][0]["text"], "Step 1")

        self.assertEqual(len(provider.messages[2]["content"]), 1)
        self.assertEqual(provider.messages[2]["content"][0]["type"], "text")
        self.assertEqual(provider.messages[2]["content"][0]["text"], "Step 2")

        # Now simulate appending the new active frame turn
        provider.messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": "Step 3"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,NEWEST"}}
            ]
        })
        self.assertEqual(len(provider.messages[-1]["content"]), 2)
        self.assertEqual(provider.messages[-1]["content"][1]["type"], "image_url")
        self.assertEqual(provider.messages[-1]["content"][1]["image_url"]["url"], "data:image/jpeg;base64,NEWEST")

    def test_run_command_action(self):
        lock = DualMonitorLock()
        handler = OSActionHandler(lock)
        
        # Test echo command
        res = handler.run_command("echo 'antiX GUI agent test'")
        self.assertIn("Exit code: 0", res)
        self.assertIn("antiX GUI agent test", res)

        # Test command in tool definitions
        openai_tools = handler.get_openai_tool_definitions()
        tool_names = [t["function"]["name"] for t in openai_tools]
        self.assertIn("run_command", tool_names)

        # Test execute_tool_by_name
        exec_res = handler.execute_tool_by_name("run_command", {"command": "echo 42"})
        self.assertIn("42", exec_res)
        self.assertIn("Exit code: 0", exec_res)

    def test_run_command_parsing(self):
        from gui_agent.agent import AgnesProvider
        lock = DualMonitorLock()
        handler = OSActionHandler(lock)

        provider = AgnesProvider(api_key="sk-fake")
        with patch.object(handler, "run_command") as mock_rc:
            json_text = '{"action": "run_command", "args": {"command": "wmctrl -l"}}'
            parsed = provider._parse_text_action(json_text, handler)
            self.assertTrue(parsed)
            mock_rc.assert_called_with("wmctrl -l")

        with patch.object(handler, "run_command") as mock_rc:
            cmd_text = 'Let\'s run command: run_command("ls -la /dev/shm")'
            parsed = provider._parse_text_action(cmd_text, handler)
            self.assertTrue(parsed)
            mock_rc.assert_called_with("ls -la /dev/shm")

    def test_ram_capture_and_jpeg_quality(self):
        import os
        from PIL import Image
        import io
        
        lock = DualMonitorLock()
        img, img_bytes = lock.capture_secondary_frame()
        self.assertIsInstance(img, Image.Image)
        self.assertGreater(len(img_bytes), 1000)
        
        test_img = Image.open(io.BytesIO(img_bytes))
        self.assertEqual(test_img.format, "JPEG")
        self.assertEqual(test_img.size, (lock.secondary.width, lock.secondary.height))

        if os.path.exists("/dev/shm/agent_frame.jpg"):
            self.assertGreater(os.path.getsize("/dev/shm/agent_frame.jpg"), 0)

    def test_human_like_dom_registration(self):
        lock = DualMonitorLock()
        handler = OSActionHandler(lock)

        with patch("subprocess.run") as mock_sub:
            mock_sub.return_value = MagicMock(returncode=0, stdout="", stderr="")
            with patch.object(handler, "snap_to_target", return_value=(200, 300)):
                with patch("time.sleep") as mock_sleep:
                    res = handler.click(200, 300)
                    self.assertIn("clicked at (200, 300)", res.lower())
                    
                    sub_calls = [c[0][0] for c in mock_sub.call_args_list]
                    mousemove_call = None
                    click_call = None
                    for call in sub_calls:
                        if "mousemove" in call:
                            mousemove_call = call
                        elif "click" in call:
                            click_call = call
                    
                    self.assertIsNotNone(mousemove_call)
                    self.assertIn("--sync", mousemove_call)
                    self.assertIsNotNone(click_call)
                    self.assertIn("1", click_call)
                    mock_sleep.assert_any_call(0.04)

    def test_vision_action_agent_fallback_chain(self):
        from gui_agent.agent import VisionActionAgent
        keystore = extract_all_keys()
        lock = DualMonitorLock()
        agent = VisionActionAgent(keystore=keystore, monitor_lock=lock, preferred_provider="auto")
        
        expected_chain = ["agnes", "groq", "bytez", "nvidia", "gemini", "openrouter"]
        
        attempts = []
        with patch("gui_agent.agent.AgnesProvider") as mock_agnes, \
             patch("gui_agent.agent.GroqProvider") as mock_groq, \
             patch("gui_agent.agent.BytezProvider") as mock_bytez, \
             patch("gui_agent.agent.NvidiaNIMProvider") as mock_nvidia, \
             patch("gui_agent.agent.GeminiProvider") as mock_gemini, \
             patch("gui_agent.agent.OpenRouterProvider") as mock_openrouter:
            
            def track_call(prov):
                attempts.append(prov)
                raise Exception(f"{prov} quota exhausted")
            
            mock_agnes.side_effect = lambda *a, **kw: track_call("agnes")
            mock_groq.side_effect = lambda *a, **kw: track_call("groq")
            mock_bytez.side_effect = lambda *a, **kw: track_call("bytez")
            mock_nvidia.side_effect = lambda *a, **kw: track_call("nvidia")
            mock_gemini.side_effect = lambda *a, **kw: track_call("gemini")
            mock_openrouter.side_effect = lambda *a, **kw: track_call("openrouter")
            
            with self.assertRaises(RuntimeError):
                agent._init_provider("test goal")
            
            self.assertEqual(attempts, [p for p in expected_chain if p in attempts])
            self.assertEqual(attempts[0], "agnes", "Primary provider in auto mode must be Agnes")


if __name__ == "__main__":
    unittest.main()
