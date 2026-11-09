# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from heretic import model as heretic_model
from heretic.config import Settings
from heretic.model import Model


class GGUFRejectionTests(unittest.TestCase):
    def test_gguf_file_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.gguf"
            model_path.touch()
            with self.assertRaises(ValueError) as context:
                heretic_model._raise_if_gguf_model(str(model_path))
            self.assertIn("GGUF", str(context.exception))
            self.assertIn("Safetensors", str(context.exception))

    def test_gguf_only_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "model-q4_k_m.gguf").touch()
            (Path(directory) / "config.json").touch()
            with self.assertRaises(ValueError) as context:
                heretic_model._raise_if_gguf_model(directory)
            self.assertIn("GGUF", str(context.exception))

    def test_directory_with_safetensors_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "model-q4_k_m.gguf").touch()
            (Path(directory) / "model.safetensors").touch()
            # Mixed repositories can be used through their Safetensors weights.
            heretic_model._raise_if_gguf_model(directory)

    def test_safetensors_only_directory_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "model.safetensors").touch()
            heretic_model._raise_if_gguf_model(directory)

    def test_empty_directory_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            heretic_model._raise_if_gguf_model(directory)

    def test_gguf_only_hub_repository_is_rejected(self):
        class Sibling:
            def __init__(self, rfilename):
                self.rfilename = rfilename

        class Info:
            siblings = [Sibling("model-q4_k_m.gguf"), Sibling("README.md")]

        with patch("heretic.model.HfApi") as api_class:
            api_class.return_value.model_info.return_value = Info()
            with self.assertRaises(ValueError) as context:
                heretic_model._raise_if_gguf_model("user/some-model-GGUF")
            self.assertIn("GGUF", str(context.exception))

    def test_model_init_rejects_gguf_with_clear_error(self):
        # Behavioral test for the user-visible failure in
        # https://github.com/p-e-w/heretic/issues/533: constructing a Model
        # from GGUF input must fail immediately with a message that names
        # the actual problem, instead of surfacing a confusing tokenizer
        # backend error from deep inside Transformers.
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "model-q4_k_m.gguf").touch()
            settings = Settings(model=directory)
            with self.assertRaises(ValueError) as context:
                Model(settings)
            message = str(context.exception)
            self.assertIn("GGUF", message)
            self.assertIn("Safetensors", message)

    def test_hub_lookup_failure_is_ignored(self):
        # Offline environments with a populated cache must still be able to
        # load models; the regular loading path reports its own errors.
        with patch("heretic.model.HfApi") as api_class:
            api_class.return_value.model_info.side_effect = OSError("offline")
            heretic_model._raise_if_gguf_model("user/some-model")


if __name__ == "__main__":
    unittest.main()
