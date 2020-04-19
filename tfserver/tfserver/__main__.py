# Copyright 2019 kubeflow.org.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import kfserving
import argparse

from tfserver import TFModel

DEFAULT_MODEL_NAME = "model"
DEFAULT_LOCAL_MODEL_DIR = "/tmp/model"

parser = argparse.ArgumentParser(parents=[kfserving.kfserver.parser])
parser.add_argument('--model_dir', required=True,
                    help='The path of the model directory')
parser.add_argument('--model_name', default=DEFAULT_MODEL_NAME,
                    help='The name that the model is served under.')
parser.add_argument('--input_name', required=True,
                    help='The model of input layer name.')
parser.add_argument('--output_name', required=True,
                    help='The name of output layer name.')
args, _ = parser.parse_known_args()

if __name__ == "__main__":
    model = TFModel(args.model_name, args.model_dir, args.input_name, args.output_name)
    model.load()
    kfserving.KFServer().start([model])
