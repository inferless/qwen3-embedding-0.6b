# Template - Deploy Qwen3-Embedding-0.6B using Inferless
Qwen team unveiled **Qwen3-Embedding-0.6B**, the smallest member of their new text-embedding and reranking line-up. Packing roughly 600 million parameters, the encoder can process up to 32k tokens and lets users select output vectors anywhere between 32 and 1024 dimensions thanks to its multi-representation-length (MRL) support. Derived from the Qwen3 foundation model, support for over 100 languages, which includes various programming languages, and provides robust multilingual, cross-lingual, and code retrieval capabilities.

Despite its modest size, Qwen3-Embedding-0.6B posts a 70.70 mean score on the multilingual MTEB leaderboard, rivaling or surpassing much larger systems such as NV-Embed-v2 (7.8 B) and GritLM-7B. It also records 61.82 on the MTEB-R reranking subset, 66.33 on the Chinese C-MTEB suite and 64.64 on the mixed-language MMTEB retrieval set, underscoring strong cross-lingual retrieval quality. 

## TL;DR:
- Deployment of Qwen3-Embedding-0.6B model using [sentence-transformers](https://github.com/UKPLab/sentence-transformers).
- Dependencies defined in `inferless-runtime-config.yaml`.
- GitHub/GitLab template creation with `app.py`, `inferless-runtime-config.yaml` and `inferless.yaml`.
- Model class in `app.py` with `initialize`, `infer`, and `finalize` functions.
- Custom runtime creation with necessary system and Python packages.
- Recommended GPU: NVIDIA A100 for optimal performance.
- Custom runtime selection in advanced configuration.
- Final review and deployment on the Inferless platform.

### Fork the Repository
Get started by forking the repository. You can do this by clicking on the fork button in the top right corner of the repository page.

This will create a copy of the repository in your own GitHub account, allowing you to make changes and customize it according to your needs.

### Create a Custom Runtime in Inferless
To access the custom runtime window in Inferless, simply navigate to the sidebar and click on the Create new Runtime button. A pop-up will appear.

Next, provide a suitable name for your custom runtime and proceed by uploading the **inferless-runtime-config.yaml** file given above. Finally, ensure you save your changes by clicking on the save button.

### Import the Model in Inferless
Log in to your inferless account, select the workspace you want the model to be imported into and click the `Add a custom model` button.

- Select `Github` as the method of upload from the Provider list and then select your Github Repository and the branch.
- Choose the type of machine, and specify the minimum and maximum number of replicas for deploying your model.
- Configure Custom Runtime ( If you have pip or apt packages), choose Volume, Secrets and set Environment variables like Inference Timeout / Container Concurrency / Scale Down Timeout
- Once you click “Continue,” click Deploy to start the model import process.

Enter all the required details to Import your model. Refer [this link](https://docs.inferless.com/integrations/git-custom-code/git--custom-code) for more information on model import.

---
## Curl Command
Following is an example of the curl command you can use to make inference. You can find the exact curl command in the Model's API page in Inferless.
```bash
curl --location '<your_inference_url>' \
    --header 'Content-Type: application/json' \
    --header 'Authorization: Bearer <your_api_key>' \
    --data '{
      "inputs": [
                  {
                        "name": "query",
                        "shape": [1],
                        "data": ["What is the capital of China?"],
                        "datatype": "BYTES"
                      },
                    {
                        "name": "document",
                        "shape": [1],
                        "data": ["The capital of China is Beijing."],
                        "datatype": "BYTES"
                      }
    ]
}'
```

---
## Customizing the Code
Open the `app.py` file. This contains the main code for inference. The `InferlessPythonModel` has three main functions, initialize, infer and finalize.

**Initialize** -  This function is executed during the cold start and is used to initialize the model. If you have any custom configurations or settings that need to be applied during the initialization, make sure to add them in this function.

**Infer** - This function is where the inference happens. The infer function leverages both RequestObjects and ResponseObjects to handle inputs and outputs in a structured and maintainable way.
- RequestObjects: Defines the input schema, validating and parsing the input data.
- ResponseObjects: Encapsulates the output data, ensuring consistent and structured API responses.

**Finalize** - This function is used to perform any cleanup activity for example you can unload the model from the gpu by setting to `None`.

For more information refer to the [Inferless docs](https://docs.inferless.com/).
