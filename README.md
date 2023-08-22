# Optimizing Cost and Operations for Private Large Language Model Inference on Azure

## Introduction
This article provides a method for private and cost-optimized deployment of large language models on the Azure cloud. It assumes the reader has foundational knowledge of large language models and Kubernetes. 

It utilizes AKS spot instances, quantization techniques, and batching inference to reduce compute costs compared to traditional deployment approaches.

A [proof-of-concept deployment](https://github.com/huangyingting/llm-inference) is presented using tools like cluster autoscaler, GPU node pools, pod affinity and tolerations to enable fast and resilient inference on low-cost spot instance GPU nodes.

### Reasons for running large language models privately
There are several motivations for organizations to run large language models privately:
- Customization - Private models can be customized for specific domains by providing proprietary training data. This improves accuracy for niche tasks like internal search or customer support..
- Confidentiality - Keeping data internal allows tighter control over sensitive information like personal data, intellectual property, or competitive intelligence.
- Control - Full control over the training data, hyperparameters, model architecture, etc. This allows you to ensure the model behaves as intended.
- Low latency - A privately hosted model can provide very low latency responses since you don't have to make API calls over the internet to external servers.
- Compliance - For regulated industries like healthcare and finance, running a private model may be important for compliance with data protection and privacy regulations.

### Challenges of running inference cost-effectively and efficiently
Deploying large language models for inference is compute-intensive, requiring high-end GPUs and specialized hardware that can be very expensive to provision on cloud platforms. For example, an 80GB Nvidia A100 GPU on Azure can generate approximately 60 completion tokens per second when running a 13 billion parameter model like Anthropic's Claude or Vicuna. If we calculate the monthly cost of running just a single A100 virtual machine in Azure's East US region, it comes out to $2,680. This one GPU could generate around 157 million tokens per month.

In contrast, conversational services like ChatGPT offered through Azure OpenAI are much more cost-effective. At Azure OpenAI ChatGPT's rate of $0.0015 per token, those 157 million tokens would only cost $236.52. Additionally, ChatGPT provides a smoother conversational experience compared to a basic 13B parameter model.

Beyond raw compute costs, the development, infrastructure, and maintenance required to build and deploy production-grade private language models at scale poses further operational challenges.

## Proof-of-concept architecture
This proof-of-concept leverages Azure Kubernetes Service (AKS), quantization, and batch inference to enable a cost-optimized private deployment of large language models while maintaining scalability and fault tolerance.
Key techniques:
- AKS enables the use of spot instances and GPUs through configurable node pools. Spot instances significantly reduce compute costs compared to on-demand instance pricing. To handle potential node evictions, AKS cluster autoscaler component monitors pod resource requirements and calculates the capacity needed for scheduled GPU workloads. It then automatically adjusts node counts to match workload demands. Considering of a 5% eviction rate, a 2 replicas GPU workalod (each replica is in a seperated node) would have at least 99.75% chance alive to serve the request. This allows the cluster to leverage cost-savings from spot instances while maintaining available capacity.
- Quantization is a technique to decrease model size and compute requirements for large language models (LLMs). It works by converting the high-precision floating point values used to represent weights and activations into lower-precision fixed-point representations that require less memory. This weight sharing through lower numeric precision allows for substantial reductions in model size and faster inference times. Some well-known quantization methods for LLMs include GGML and GPT-Q. 
- Batching combines multiple inference requests into a batch call instead of handling requests one by one, improving utilization for intermittent loads. This technique is especially useful for LLMs since they are compute-intensive and can take several seconds to complete a single inference request. vllm and text-generation-inference are two typical frameworks of batch inference for LLMs. In next proof-of-concept deployment section, we will use vllm to demonstrate the benefits of batching.

![Architecture](https://raw.githubusercontent.com/huangyingting/llm-inference/main/docs/images/LLM-AKS.svg)

By combining these techniques, good speedup can be obtained while maintaining model accuracy, thus improving actual inference latency and throughput of LLMs with affordable and scalable private deployment.

## Proof-of-concept deployment and considerations
Before we begin, please make sure you have the following prerequisites:
- Have an existing AKS cluster. If you don't have a cluster, create one using the [Azure CLI](https://learn.microsoft.com/en-us/azure/aks/learn/quick-kubernetes-deploy-cli), [Azure PowerShell](https://learn.microsoft.com/en-us/azure/aks/learn/quick-kubernetes-deploy-powershell), or the [Azure portal](https://learn.microsoft.com/en-us/azure/aks/learn/quick-kubernetes-deploy-portal).
- Have `aks-preview` extension installed and `GPUDedicatedVHDPreview` feature registered. You can follow the [instructions](https://docs.microsoft.com/en-us/azure/aks/gpu-cluster#enable-the-gpu-dedicated-vhd-preview-feature) to install the extension and register the feature.
- Have a container image for your LLM inference. For PoC purpose we use [vllm](https://github.com/vllm-project/vllm) as the inference framework and [this image](ghcr.io/huangyingting/llm-inference-vllm:main), the image is built from [this Dockerfile](https://github.com/huangyingting/llm-inference/vllm/Dockerfile)

**NOTE**: It is suggested to build your own image for production use. You have full control over all dependencies and compatibility between GPU accelerated application and GPU driver installed on GPU node. For more information, please refer to [CUDA Compatibility and Upgrades](https://docs.nvidia.com/deploy/cuda-compatibility/index.html).

### Define environment variables
We need to define the following environment variables for creating GPU spot node pool. You can find the values for these variables in the Azure portal or by using the Azure CLI.

```shell
# AKS cluster info
export RESOURCE_GROUP_NAME=your_aks_resource_group_name
export CLUSTER_NAME=your_aks_cluster_name
export REGION=your_aks_cluster_region
# GPU VM size
export VM_SIZE=Standard_NC4as_T4_v3
```

### Add a GPU spot node pool
Now you can add a spot node pool for GPU nodes into existing AKS cluster. The command in below will create a new node pool named `gpunp` with 1 node of `Standard_NC4as_T4_v3` size. The node pool will be configured with GPU taints and cluster autoscaler. The GPU taints will ensure that only GPU workloads are scheduled on the node pool. The cluster autoscaler will automatically adjust the number of nodes in the node pool based on the workload demands. The node pool will also be configured with spot instances to reduce compute costs. The spot-max-price is set to -1 to ensure that the node pool will not be evicted due to price changes. The min-count and max-count are set to 1 to ensure that the node pool will always have at least 1 node available for scheduling.

```shell
az aks nodepool add \
    --resource-group $RESOURCE_GROUP_NAME \
    --cluster-name $CLUSTER_NAME \
    --name gpunp \
    --node-count 1 \
    --node-vm-size $VM_SIZE \
    --node-taints sku=gpu:NoSchedule \
    --aks-custom-headers UseGPUDedicatedVHD=true \
    --enable-cluster-autoscaler \
    --priority Spot \
    --eviction-policy Delete \
    --spot-max-price -1 \
    --min-count 1 \
    --max-count 2
```

### Deploy LLM inference service
Now you can deploy the LLM inference service to the AKS cluster. The deployment manifests are located in [this place](./vllm/manifests/), there are two manifests, each represents a different storage backend for storing the model files:

`vllm-azure-disk.yaml` will deploy a `StatefulSet` with 2 replicas and a Service for the LLM inference service. The StatefulSet will be configured with pod anti-affinity to ensure that the replicas are scheduled on different nodes. The StatefulSet will also be configured with tolerations to ensure that the replicas are scheduled on the GPU spot node pool. The Service will be configured with a ClusterIP to expose the LLM inference service on the AKS cluster. Each replica will be configured with a PersistentVolumeClaim to mount a 16GB Azure Disk for storing the model files.

`vllm-azure-files.yaml` will deploy a `Deployment` with 2 replicas and a Service for the LLM inference service. The StatefulSet will be configured with pod anti-affinity to ensure that the replicas are scheduled on different nodes. The StatefulSet will also be configured with tolerations to ensure that the replicas are scheduled on the GPU spot node pool. The Service will be configured with a ClusterIP to expose the LLM inference service on the AKS cluster. Each replica will be configured with a PersistentVolumeClaim to mount a 16GB Azure File Share for storing the model files, the persistent volume is shared between replicas.

AKS also supports shared disk, however, it only supports block volume, Multi-node read write is not supported by common file systems (e.g. ext4, xfs), it's only supported by cluster file systems. So we can't use shared disk for storing the model files.

AKS will taints the GPU spot nodes with `nvidia.com/gpu:NoSchedule`, `sku=gpu:NoSchedule` and `kubernetes.azure.com/scalesetpriority=spot:NoSchedule`. To schedule GPU workloads on spot instance nodes, we need to add tolerations to the pod spec. This allows the pods to be scheduled on nodes tainted by AKS for GPU spot nodes.

We can create a namespace and add default tolerations to it. This ensures all pods in the namespace can be scheduled on GPU spot nodes.

The following example creates the 'llm' namespace and adds the required tolerations:

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: llm
  annotations:
    scheduler.alpha.kubernetes.io/defaultTolerations: '[{"Key": "kubernetes.azure.com/scalesetpriority", "Operator": "Equal", "Value": "spot", "Effect": "NoSchedule"}, {"Key": "sku", "Operator": "Equal", "Value": "gpu", "Effect": "NoSchedule"}]'
```

Now any pods created in the 'llm' namespace will have the tolerations needed to schedule on spot instance nodes with GPUs. This provides an easy way to deploy GPU workloads on spot instances.

To start the deployment, run the following command:
```shell
kubeclt apply -f vllm/manifests/vllm-azure-disk.yaml
```
**NOTE**: The manifests are configured to use the image `ghcr.io/huangyingting/llm-inference-vllm:main`, you can change it to your own image.


### AKS cluster autoscaler
We deployed two replicas, however, the GPU node only has 1 node, cluster autoscaler noticed that and scale up the node pool to 2 nodes. You can check the status of cluster autoscaler by running the following command:

```shell
kubectl describe pod vllm-1 -n llm

Name:             vllm-1
Namespace:        llm
Priority:         0
...
Node-Selectors:              <none>
Tolerations:                 kubernetes.azure.com/scalesetpriority=spot:NoSchedule
                             node.kubernetes.io/not-ready:NoExecute op=Exists for 300s
                             node.kubernetes.io/unreachable:NoExecute op=Exists for 300s
                             nvidia.com/gpu:NoSchedule op=Exists
                             sku=gpu:NoSchedule
Events:
  Type     Reason                  Age                    From                     Message
  ----     ------                  ----                   ----                     -------
  Normal   NotTriggerScaleUp       6m42s                  cluster-autoscaler       pod didn't trigger scale-up: 2 max node group size reached
  Normal   TriggeredScaleUp        3m5s                   cluster-autoscaler       pod triggered scale-up: [{aks-gpunp-98036146-vmss 1->2 (max: 2)}]
  Warning  FailedScheduling        2m12s (x2 over 7m14s)  default-scheduler        0/2 nodes are available: 2 Insufficient nvidia.com/gpu. preemption: 0/2 nodes are available: 2 No preemption victims found for incoming pod..
  Warning  FailedScheduling        34s                    default-scheduler        0/3 nodes are available: 1 node(s) had untolerated taint {node.kubernetes.io/network-unavailable: }, 2 Insufficient nvidia.com/gpu. preemption: 0/3 nodes are available: 1 Preemption is not helpful for scheduling, 2 No preemption victims found for incoming pod..
  Normal   Scheduled               19s                    default-scheduler        Successfully assigned llm/vllm-1 to aks-gpunp-98036146-vmss000004
  Normal   SuccessfulAttachVolume  9s                     attachdetach-controller  AttachVolume.Attach succeeded for volume "pvc-d3ccf681-99ae-411a-8d72-33063bf9e422"
  Normal   Pulling                 4s                     kubelet                  Pulling image "ghcr.io/huangyingting/llm-inference-vllm:main"
```

We can also trigger node eviction by issuing the following command, this will simulate a node eviction and trigger the cluster autoscaler to scale up the node pool:

```shell
az vmss simulate-eviction --resource-group MC_$RESOURCE_GROUP_NAME_$CLUSTER_NAME_$REGION --name $NODE_POOL --instance-id 0
```

### Test LLM inference
The deployment comes with a default model `facebook/opt-125m` and an OpenAI API compatiable endpoint. You can test the inference service by running the following command:

```shell
# port forwarding to the service
kubeclt port-forward svc/vllm 9090:8080 -n llm
```

```shell
curl http://localhost:9090/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "facebook/opt-125m",
        "prompt": "San Francisco is a",
        "max_tokens": 7,
        "temperature": 0
    }'
```

## Wrap-up
In summary, this article has outlined an approach to deploy large language models on Azure Kubernetes Service in a cost-optimized and scalable manner for private inference. By combining cloud-native techniques like spot instances and autoscaling with model optimization methods like quantization and batching, organizations can unlock the advanced AI capabilities of large models while maintaining control, low latency, and reduced costs compared to public cloud services. 
