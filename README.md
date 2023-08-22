# Optimizing Cost and Operations for Private Large Language Model Inference on Azure

## Introduction
This article provides a method for private and cost-optimized deployment of large language models on the Azure cloud. It assumes the reader has foundational knowledge of large language models and Kubernetes. The proposed solution utilizes Azure Kubernetes Service along with open source projects for fast inference to build a proof-of-concept for affordable large language model hosting.

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

![Architecture](docs/images/LLM-AKS.svg "Architecture")

By combining these techniques, good speedup can be obtained while maintaining model accuracy, thus improving actual inference latency and throughput of LLMs with affordable and scalable private deployment.

## Proof-of-concept deployment and considerations
Before you begin, please make sure you have the following prerequisites:
- Have an existing AKS cluster. If you don't have a cluster, create one using the [Azure CLI](https://learn.microsoft.com/en-us/azure/aks/learn/quick-kubernetes-deploy-cli), [Azure PowerShell](https://learn.microsoft.com/en-us/azure/aks/learn/quick-kubernetes-deploy-powershell), or the [Azure portal](https://learn.microsoft.com/en-us/azure/aks/learn/quick-kubernetes-deploy-portal).
- Have `aks-preview` extension installed and `GPUDedicatedVHDPreview` feature registered. You can follow the [instructions](https://docs.microsoft.com/en-us/azure/aks/gpu-cluster#enable-the-gpu-dedicated-vhd-preview-feature) to install the extension and register the feature.
- Have a container image for your LLM inference. For PoC purpose we use [vllm](https://github.com/vllm-project/vllm) as the inference framework and [this image](ghcr.io/huangyingting/inference-images-vllm:main), the image is built from [this Dockerfile](./vllm/Dockerfile)

### Define environment variables
First, define the following environment variables for your AKS cluster. You can find the values for these variables in the Azure portal or by using the Azure CLI.

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
    --max-count 1
```

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: llm
  annotations:
    scheduler.alpha.kubernetes.io/defaultTolerations: '[{"Key": "kubernetes.azure.com/scalesetpriority", "Operator": "Equal", "Value": "spot", "Effect": "NoSchedule"}, {"Key": "sku", "Operator": "Equal", "Value": "gpu", "Effect": "NoSchedule"}]'
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  labels:
    app: vllm
  name: vllm
  namespace: llm
spec:
  selector:
    matchLabels:
      app: vllm
  serviceName: vllm
  replicas: 2
  template:
    metadata:
      labels:
        app: vllm
    spec:
      affinity:
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector:
              matchExpressions:
                - key: "app"
                  operator: In
                  values:
                  - vllm
            topologyKey: "kubernetes.io/hostname"    
      containers:
      - image: ghcr.io/huangyingting/inference-images-vllm:main
        imagePullPolicy: Always
        name: vllm
        resources:
          limits:
           nvidia.com/gpu: 1        
        ports:
        - containerPort: 8080
          name: http
          protocol: TCP        
        volumeMounts:
        - name: shm
          mountPath: /dev/shm        
        - name: vllm
          mountPath: "/data"
      volumes:
      - name: shm
        emptyDir:
          medium: Memory
          sizeLimit: 1Gi
  volumeClaimTemplates:
  - metadata:
      name: vllm
    spec:
      accessModes: [ "ReadWriteOnce" ]
      storageClassName: managed-csi
      resources:
        requests:
          storage: 16Gi      
---
apiVersion: v1
kind: Service
metadata:
  name: vllm
  namespace: llm
  labels:
    app: vllm
spec:
  ports:
  - port: 8080
    targetPort: 8000
  selector:
    app: vllm
```

```shell
az vmss simulate-eviction --resource-group MC_$RESOURCE_GROUP_NAME_$CLUSTER_NAME_$REGION --name $NODE_POOL --instance-id 0
```

```shell
k describe pod vllm-1 -n llm
Name:             vllm-1
Namespace:        llm
Priority:         0
Service Account:  default
Node:             aks-gpunp-98036146-vmss000004/10.224.0.6
Start Time:       Mon, 21 Aug 2023 05:40:20 +0000
Labels:           app=vllm
                  controller-revision-hash=vllm-54b57d668
                  statefulset.kubernetes.io/pod-name=vllm-1
Annotations:      <none>
Status:           Pending
IP:               
IPs:              <none>
Controlled By:    StatefulSet/vllm
Containers:
  vllm:
    Container ID:   
    Image:          ghcr.io/huangyingting/inference-images-vllm:main
    Image ID:       
    Port:           8080/TCP
    Host Port:      0/TCP
    State:          Waiting
      Reason:       ContainerCreating
    Ready:          False
    Restart Count:  0
    Limits:
      nvidia.com/gpu:  1
    Requests:
      nvidia.com/gpu:  1
    Environment:       <none>
    Mounts:
      /data from vllm-models-disk (rw)
      /dev/shm from shm (rw)
      /var/run/secrets/kubernetes.io/serviceaccount from kube-api-access-pn6gb (ro)
Conditions:
  Type              Status
  Initialized       True 
  Ready             False 
  ContainersReady   False 
  PodScheduled      True 
Volumes:
  vllm-models-disk:
    Type:       PersistentVolumeClaim (a reference to a PersistentVolumeClaim in the same namespace)
    ClaimName:  vllm-models-disk-vllm-1
    ReadOnly:   false
  shm:
    Type:       EmptyDir (a temporary directory that shares a pod's lifetime)
    Medium:     Memory
    SizeLimit:  1Gi
  kube-api-access-pn6gb:
    Type:                    Projected (a volume that contains injected data from multiple sources)
    TokenExpirationSeconds:  3607
    ConfigMapName:           kube-root-ca.crt
    ConfigMapOptional:       <nil>
    DownwardAPI:             true
QoS Class:                   BestEffort
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
  Normal   Pulling                 4s                     kubelet                  Pulling image "ghcr.io/huangyingting/inference-images-vllm:main"
```

```shell
kubeclt port-forward svc/vllm 9090:8080 -n llm
```

```shell
curl --location 'http://localhost:9090/models/apply' \
--header 'Content-Type: application/json' \
--data-raw '{
    "id": "TheBloke/vicuna-7B-v1.5-GGML/vicuna-7b-v1.5.ggmlv3.q4_0.bin",
    "name": "vicuna-7b-v1.5"
}'
```

```shell
cat <<EOF > vicuna-7b-v1.5.yaml 
backend: llama
context_size: 2000
f16: true 
gpu_layers: 43
low_vram: false
mmap: true
mmlock: true
batch: 512
name: vicuna-7b-v1.5
parameters:
  model: vicuna-7b-v1.5.ggmlv3.q4_0.bin
  temperature: 0.2
  top_k: 80
  top_p: 0.7
roles:
  assistant: '### Response:'
  system: '### System:'
  user: '### Instruction:'
template:
  chat: vicuna-chat
  completion: vicuna-completion
EOF
```

```shell
kubectl delete pod --all -n llm
```

```shell
curl http://localhost:9090/v1/chat/completions -H "Content-Type: application/json" -d '{
     "model": "vicuna-7b-v1.5",
     "messages": [{"role": "user", "content": "How are you?"}],
     "temperature": 0.9 
   }'
```