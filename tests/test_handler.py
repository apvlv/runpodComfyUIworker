import unittest
from unittest.mock import patch, MagicMock, Mock
import sys
import os
import json
import base64

# Make sure that the root is in path so we can import handler.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import handler

# Local folder for test resources
RUNPOD_WORKER_COMFY_TEST_RESOURCES_IMAGES = "./test_resources/images"


class TestValidateInput(unittest.TestCase):
    """Tests for the validate_input function."""

    def test_valid_input_with_workflow_only(self):
        input_data = {"workflow": {"key": "value"}}
        validated_data, error = handler.validate_input(input_data)
        self.assertIsNone(error)
        self.assertEqual(
            validated_data,
            {"workflow": {"key": "value"}, "images": None, "comfy_org_api_key": None},
        )

    def test_valid_input_with_workflow_and_images(self):
        input_data = {
            "workflow": {"key": "value"},
            "images": [{"name": "image1.png", "image": "base64string"}],
        }
        validated_data, error = handler.validate_input(input_data)
        self.assertIsNone(error)
        expected = {
            "workflow": {"key": "value"},
            "images": [{"name": "image1.png", "image": "base64string"}],
            "comfy_org_api_key": None,
        }
        self.assertEqual(validated_data, expected)

    def test_valid_input_with_comfy_org_api_key(self):
        input_data = {
            "workflow": {"key": "value"},
            "comfy_org_api_key": "test-api-key-123",
        }
        validated_data, error = handler.validate_input(input_data)
        self.assertIsNone(error)
        expected = {
            "workflow": {"key": "value"},
            "images": None,
            "comfy_org_api_key": "test-api-key-123",
        }
        self.assertEqual(validated_data, expected)

    def test_input_missing_workflow(self):
        input_data = {"images": [{"name": "image1.png", "image": "base64string"}]}
        validated_data, error = handler.validate_input(input_data)
        self.assertIsNotNone(error)
        self.assertEqual(error, "Missing 'workflow' parameter")

    def test_input_with_invalid_images_structure(self):
        input_data = {
            "workflow": {"key": "value"},
            "images": [{"name": "image1.png"}],  # Missing 'image' key
        }
        validated_data, error = handler.validate_input(input_data)
        self.assertIsNotNone(error)
        self.assertEqual(
            error, "'images' must be a list of objects with 'name' and 'image' keys"
        )

    def test_invalid_json_string_input(self):
        input_data = "invalid json"
        validated_data, error = handler.validate_input(input_data)
        self.assertIsNotNone(error)
        self.assertEqual(error, "Invalid JSON format in input")

    def test_valid_json_string_input(self):
        input_data = '{"workflow": {"key": "value"}}'
        validated_data, error = handler.validate_input(input_data)
        self.assertIsNone(error)
        self.assertEqual(
            validated_data,
            {"workflow": {"key": "value"}, "images": None, "comfy_org_api_key": None},
        )

    def test_empty_input(self):
        input_data = None
        validated_data, error = handler.validate_input(input_data)
        self.assertIsNotNone(error)
        self.assertEqual(error, "Please provide input")


class TestCheckServer(unittest.TestCase):
    """Tests for the check_server function."""

    @patch("handler.requests.get")
    def test_check_server_server_up(self, mock_requests):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_requests.return_value = mock_response

        result = handler.check_server("http://127.0.0.1:8188", 1, 50)
        self.assertTrue(result)

    @patch("handler.requests.get")
    def test_check_server_server_down(self, mock_requests):
        mock_requests.side_effect = handler.requests.RequestException()
        result = handler.check_server("http://127.0.0.1:8188", 1, 50)
        self.assertFalse(result)

    @patch("handler.requests.get")
    def test_check_server_timeout(self, mock_requests):
        mock_requests.side_effect = handler.requests.Timeout()
        result = handler.check_server("http://127.0.0.1:8188", 1, 50)
        self.assertFalse(result)


class TestQueueWorkflow(unittest.TestCase):
    """Tests for the queue_workflow function."""

    @patch("handler.requests.post")
    def test_queue_workflow_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"prompt_id": "123"}
        mock_post.return_value = mock_response

        result = handler.queue_workflow({"prompt": "test"}, "client-id-123")
        self.assertEqual(result, {"prompt_id": "123"})

    @patch("handler.requests.post")
    @patch("handler.get_available_models")
    def test_queue_workflow_validation_error(self, mock_get_models, mock_post):
        mock_get_models.return_value = {"checkpoints": ["model1.safetensors"]}
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = '{"error": "validation failed"}'
        mock_response.json.return_value = {"error": "validation failed"}
        mock_post.return_value = mock_response

        with self.assertRaises(ValueError):
            handler.queue_workflow({"prompt": "test"}, "client-id-123")

    @patch("handler.requests.post")
    def test_queue_workflow_with_api_key(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"prompt_id": "123"}
        mock_post.return_value = mock_response

        result = handler.queue_workflow(
            {"prompt": "test"}, "client-id-123", comfy_org_api_key="test-key"
        )

        # Verify the API key was included in the payload
        call_args = mock_post.call_args
        payload = json.loads(call_args.kwargs["data"])
        self.assertEqual(payload["extra_data"]["api_key_comfy_org"], "test-key")
        self.assertEqual(result, {"prompt_id": "123"})


class TestGetHistory(unittest.TestCase):
    """Tests for the get_history function."""

    @patch("handler.requests.get")
    def test_get_history_success(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {"key": "value"}
        mock_get.return_value = mock_response

        result = handler.get_history("123")

        self.assertEqual(result, {"key": "value"})
        mock_get.assert_called_with(
            "http://127.0.0.1:8188/history/123", timeout=30
        )


class TestGetImageData(unittest.TestCase):
    """Tests for the get_image_data function."""

    @patch("handler.requests.get")
    def test_get_image_data_success(self, mock_get):
        mock_response = MagicMock()
        mock_response.content = b"image_bytes"
        mock_get.return_value = mock_response

        result = handler.get_image_data("test.png", "subfolder", "output")

        self.assertEqual(result, b"image_bytes")

    @patch("handler.requests.get")
    def test_get_image_data_timeout(self, mock_get):
        mock_get.side_effect = handler.requests.Timeout()

        result = handler.get_image_data("test.png", "subfolder", "output")

        self.assertIsNone(result)

    @patch("handler.requests.get")
    def test_get_image_data_request_error(self, mock_get):
        mock_get.side_effect = handler.requests.RequestException("error")

        result = handler.get_image_data("test.png", "subfolder", "output")

        self.assertIsNone(result)


class TestUploadImages(unittest.TestCase):
    """Tests for the upload_images function."""

    @patch("handler.requests.post")
    def test_upload_images_successful(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        test_image_data = base64.b64encode(b"Test Image Data").decode("utf-8")
        images = [{"name": "test_image.png", "image": test_image_data}]

        result = handler.upload_images(images)

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["message"], "All images uploaded successfully")

    @patch("handler.requests.post")
    def test_upload_images_with_data_uri_prefix(self, mock_post):
        """Test that data URI prefix is properly stripped."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        test_image_data = base64.b64encode(b"Test Image Data").decode("utf-8")
        images = [
            {"name": "test_image.png", "image": f"data:image/png;base64,{test_image_data}"}
        ]

        result = handler.upload_images(images)

        self.assertEqual(result["status"], "success")

    @patch("handler.requests.post")
    def test_upload_images_failed(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.raise_for_status.side_effect = handler.requests.HTTPError("Error")
        mock_post.return_value = mock_response

        test_image_data = base64.b64encode(b"Test Image Data").decode("utf-8")
        images = [{"name": "test_image.png", "image": test_image_data}]

        result = handler.upload_images(images)

        self.assertEqual(result["status"], "error")

    def test_upload_images_empty_list(self):
        result = handler.upload_images([])

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["message"], "No images to upload")

    def test_upload_images_none(self):
        result = handler.upload_images(None)

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["message"], "No images to upload")

    @patch("handler.requests.post")
    def test_upload_images_invalid_base64(self, mock_post):
        """Test handling of invalid base64 data."""
        images = [{"name": "test_image.png", "image": "not-valid-base64!@#$"}]

        result = handler.upload_images(images)

        self.assertEqual(result["status"], "error")


class TestGetAvailableModels(unittest.TestCase):
    """Tests for the get_available_models function."""

    @patch("handler.requests.get")
    def test_get_available_models_success(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "CheckpointLoaderSimple": {
                "input": {
                    "required": {
                        "ckpt_name": [["model1.safetensors", "model2.safetensors"]]
                    }
                }
            }
        }
        mock_get.return_value = mock_response

        result = handler.get_available_models()

        self.assertEqual(
            result["checkpoints"], ["model1.safetensors", "model2.safetensors"]
        )

    @patch("handler.requests.get")
    def test_get_available_models_error(self, mock_get):
        mock_get.side_effect = Exception("Network error")

        result = handler.get_available_models()

        self.assertEqual(result, {})


class TestComfyServerStatus(unittest.TestCase):
    """Tests for the _comfy_server_status function."""

    @patch("handler.requests.get")
    def test_comfy_server_status_reachable(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        result = handler._comfy_server_status()

        self.assertTrue(result["reachable"])
        self.assertEqual(result["status_code"], 200)

    @patch("handler.requests.get")
    def test_comfy_server_status_unreachable(self, mock_get):
        mock_get.side_effect = Exception("Connection refused")

        result = handler._comfy_server_status()

        self.assertFalse(result["reachable"])
        self.assertIn("error", result)


class TestAttemptWebsocketReconnect(unittest.TestCase):
    """Tests for the _attempt_websocket_reconnect function."""

    @patch("handler._comfy_server_status")
    @patch("handler.websocket.WebSocket")
    def test_reconnect_success(self, mock_ws_class, mock_server_status):
        mock_server_status.return_value = {"reachable": True, "status_code": 200}
        mock_ws_instance = MagicMock()
        mock_ws_class.return_value = mock_ws_instance

        result = handler._attempt_websocket_reconnect(
            "ws://127.0.0.1:8188/ws?clientId=test", 3, 0, Exception("initial")
        )

        self.assertEqual(result, mock_ws_instance)
        mock_ws_instance.connect.assert_called_once()

    @patch("handler._comfy_server_status")
    @patch("handler.websocket.WebSocket")
    def test_reconnect_server_unreachable(self, mock_ws_class, mock_server_status):
        mock_server_status.return_value = {"reachable": False, "error": "connection refused"}

        with self.assertRaises(handler.websocket.WebSocketConnectionClosedException):
            handler._attempt_websocket_reconnect(
                "ws://127.0.0.1:8188/ws?clientId=test", 3, 0, Exception("initial")
            )

    @patch("handler._comfy_server_status")
    @patch("handler.websocket.WebSocket")
    @patch("handler.time.sleep")
    def test_reconnect_retry_then_success(self, mock_sleep, mock_ws_class, mock_server_status):
        mock_server_status.return_value = {"reachable": True, "status_code": 200}
        mock_ws_instance = MagicMock()
        mock_ws_class.return_value = mock_ws_instance
        # First attempt fails, second succeeds
        mock_ws_instance.connect.side_effect = [
            handler.websocket.WebSocketException("fail"),
            None,
        ]

        result = handler._attempt_websocket_reconnect(
            "ws://127.0.0.1:8188/ws?clientId=test", 3, 1, Exception("initial")
        )

        self.assertEqual(result, mock_ws_instance)
        self.assertEqual(mock_ws_instance.connect.call_count, 2)
        mock_sleep.assert_called_once_with(1)

    @patch("handler._comfy_server_status")
    @patch("handler.websocket.WebSocket")
    @patch("handler.time.sleep")
    def test_reconnect_all_attempts_fail(self, mock_sleep, mock_ws_class, mock_server_status):
        mock_server_status.return_value = {"reachable": True, "status_code": 200}
        mock_ws_instance = MagicMock()
        mock_ws_class.return_value = mock_ws_instance
        mock_ws_instance.connect.side_effect = handler.websocket.WebSocketException("fail")

        with self.assertRaises(handler.websocket.WebSocketConnectionClosedException):
            handler._attempt_websocket_reconnect(
                "ws://127.0.0.1:8188/ws?clientId=test", 3, 0, Exception("initial")
            )

        self.assertEqual(mock_ws_instance.connect.call_count, 3)


class TestHandlerFunction(unittest.TestCase):
    """Tests for the main handler function."""

    @patch("handler.check_server")
    @patch("handler.validate_input")
    def test_handler_invalid_input(self, mock_validate, mock_check_server):
        mock_validate.return_value = (None, "Missing 'workflow' parameter")

        job = {"id": "test-job-123", "input": {}}
        result = handler.handler(job)

        self.assertEqual(result, {"error": "Missing 'workflow' parameter"})

    @patch("handler.check_server")
    @patch("handler.validate_input")
    def test_handler_server_unavailable(self, mock_validate, mock_check_server):
        mock_validate.return_value = ({"workflow": {}, "images": None, "comfy_org_api_key": None}, None)
        mock_check_server.return_value = False

        job = {"id": "test-job-123", "input": {"workflow": {}}}
        result = handler.handler(job)

        self.assertIn("error", result)
        self.assertIn("not reachable", result["error"])

    @patch("handler.check_server")
    @patch("handler.validate_input")
    @patch("handler.upload_images")
    def test_handler_upload_images_failure(self, mock_upload, mock_validate, mock_check_server):
        mock_validate.return_value = (
            {"workflow": {}, "images": [{"name": "test.png", "image": "data"}], "comfy_org_api_key": None},
            None,
        )
        mock_check_server.return_value = True
        mock_upload.return_value = {"status": "error", "details": ["upload failed"]}

        job = {"id": "test-job-123", "input": {"workflow": {}, "images": [{"name": "test.png", "image": "data"}]}}
        result = handler.handler(job)

        self.assertIn("error", result)
        self.assertEqual(result["error"], "Failed to upload one or more input images")


if __name__ == "__main__":
    unittest.main()
