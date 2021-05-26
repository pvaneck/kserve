/*
Copyright 2020 kubeflow.org.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package components

import (
	"context"
	"fmt"
	"strconv"
	"strings"

	"github.com/go-logr/logr"
	"github.com/kubeflow/kfserving/pkg/constants"
	"github.com/kubeflow/kfserving/pkg/controller/v1alpha1/trainedmodel/sharding/memory"
	v1beta1utils "github.com/kubeflow/kfserving/pkg/controller/v1beta1/inferenceservice/utils"
	"github.com/kubeflow/kfserving/pkg/credentials"
	"github.com/pkg/errors"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/rest"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"github.com/kubeflow/kfserving/pkg/apis/serving/v1beta1"
)

var _ Component = &Predictor{}

// Predictor reconciles resources for this component.
type Predictor struct {
	client                 client.Client
	scheme                 *runtime.Scheme
	inferenceServiceConfig *v1beta1.InferenceServicesConfig
	credentialBuilder      *credentials.CredentialBuilder
	Log                    logr.Logger
}

func NewPredictor(client client.Client, scheme *runtime.Scheme, inferenceServiceConfig *v1beta1.InferenceServicesConfig) Component {
	return &Predictor{
		client:                 client,
		scheme:                 scheme,
		inferenceServiceConfig: inferenceServiceConfig,
		Log:                    ctrl.Log.WithName("PredictorReconciler"),
	}
}

// Reconcile observes the predictor and attempts to drive the status towards the desired state.
// func (p *Predictor) Reconcile(isvc *v1beta1.InferenceService) error {
// 	p.Log.Info("!-----------! Reconciling Predictor", "PredictorSpec", isvc.Spec.Predictor)
// 	predictor := isvc.Spec.Predictor.GetImplementation()
// 	out, err := json.MarshalIndent(isvc.Spec.Predictor, "", "  ")
// 	p.Log.Info("!--^^^^^^--! Predictor Implementation", string(out), "+++++", string(v1beta1.PredictorComponent))
// 	p.Log.Info("!--^^^^^^--! Predictor 2---", reflect.TypeOf(predictor), "+++++", string(v1beta1.PredictorComponent))
// 	annotations := utils.Filter(isvc.Annotations, func(key string) bool {
// 		return !utils.Includes(constants.ServiceAnnotationDisallowedList, key)
// 	})
// 	// KNative does not support INIT containers or mounting, so we add annotations that trigger the
// 	// StorageInitializer injector to mutate the underlying deployment to provision model data
// 	if sourceURI := predictor.GetStorageUri(); sourceURI != nil {
// 		annotations[constants.StorageInitializerSourceUriInternalAnnotationKey] = *sourceURI
// 	}
// 	hasInferenceLogging := addLoggerAnnotations(isvc.Spec.Predictor.Logger, annotations)
// 	hasInferenceBatcher := addBatcherAnnotations(isvc.Spec.Predictor.Batcher, annotations)
// 	// Add agent annotations so mutator will mount model agent to multi-model InferenceService's predictor
// 	addAgentAnnotations(isvc, annotations, p.inferenceServiceConfig)

// 	objectMeta := metav1.ObjectMeta{
// 		Name:      constants.DefaultPredictorServiceName(isvc.Name),
// 		Namespace: isvc.Namespace,
// 		Labels: utils.Union(isvc.Labels, map[string]string{
// 			constants.InferenceServicePodLabelKey: isvc.Name,
// 			constants.KServiceComponentLabel:      string(v1beta1.PredictorComponent),
// 		}),
// 		Annotations: annotations,
// 	}
// 	container := predictor.GetContainer(isvc.ObjectMeta, isvc.Spec.Predictor.GetExtensions(), p.inferenceServiceConfig)
// 	if len(isvc.Spec.Predictor.PodSpec.Containers) == 0 {
// 		isvc.Spec.Predictor.PodSpec.Containers = []v1.Container{
// 			*container,
// 		}
// 	} else {
// 		isvc.Spec.Predictor.PodSpec.Containers[0] = *container
// 	}
// 	//TODO now knative supports multi containers, consolidate logger/batcher/puller to the sidecar container
// 	//https://github.com/kubeflow/kfserving/issues/973
// 	if hasInferenceLogging {
// 		addLoggerContainerPort(&isvc.Spec.Predictor.PodSpec.Containers[0])
// 	}

// 	if hasInferenceBatcher {
// 		addBatcherContainerPort(&isvc.Spec.Predictor.PodSpec.Containers[0])
// 	}

// 	podSpec := v1.PodSpec(isvc.Spec.Predictor.PodSpec)

// 	// Reconcile modelConfig
// 	configMapReconciler := modelconfig.NewModelConfigReconciler(p.client, p.scheme)
// 	if err := configMapReconciler.Reconcile(isvc); err != nil {
// 		return err
// 	}

// 	// Here we allow switch between knative and vanilla deployment
// 	r := knative.NewKsvcReconciler(p.client, p.scheme, objectMeta, &isvc.Spec.Predictor.ComponentExtensionSpec,
// 		&podSpec, isvc.Status.Components[v1beta1.PredictorComponent])

// 	if err := controllerutil.SetControllerReference(isvc, r.Service, p.scheme); err != nil {
// 		return errors.Wrapf(err, "fails to set owner reference for predictor")
// 	}
// 	status, err := r.Reconcile()
// 	if err != nil {
// 		return errors.Wrapf(err, "fails to reconcile predictor")
// 	}
// 	isvc.Status.PropagateStatus(v1beta1.PredictorComponent, status)
// 	return nil
// }

func getPredictorFramework(predictorSpec *v1beta1.PredictorSpec) (string, error) {
	if predictorSpec.XGBoost != nil {
		return "xgboost", nil
	}
	if predictorSpec.LightGBM != nil {
		return "lightgbm", nil
	}
	if predictorSpec.SKLearn != nil {
		return "sklearn", nil
	}
	if predictorSpec.Tensorflow != nil {
		return "tensorflow", nil
	}
	if predictorSpec.ONNX != nil {
		return "onnx", nil
	}
	if predictorSpec.PyTorch != nil {
		return "pytorch", nil
	}
	if predictorSpec.Triton != nil {
		return "triton", nil
	}
	if predictorSpec.PMML != nil {
		return "pmml", nil
	}

	return "", errors.New("Valid predictor framework not found.")
}

func (p *Predictor) Reconcile(isvc *v1beta1.InferenceService) error {
	p.Log.Info("!-----------! Reconciling Predictor", "PredictorSpec", isvc.Spec.Predictor)
	predictor := isvc.Spec.Predictor.GetImplementation()

	framework, err := getPredictorFramework(&isvc.Spec.Predictor)
	if err != nil {
		p.Log.Error(err, "Failed to get valid framework.")
		return nil
	}
	p.Log.Info("!!! ------", "Framework", framework)
	sourceURI := predictor.GetStorageUri()

	serviceAccountName := isvc.Spec.Predictor.ServiceAccountName
	if serviceAccountName == "" {
		return errors.New("No secret key specified under serviceAccountName")
	}

	p.Log.Info("!-Source URI -------------!", "SourceURI", sourceURI)
	if sourceURI == nil {
		return errors.New("No model specified with storageUri")
	}

	if !strings.HasPrefix(*sourceURI, "s3://") {
		return errors.New("Only s3 storageUri supported")
	}

	runtimeVersion := predictor.GetRuntimeVersion()
	p.Log.Info("!-- Runtime version -------------!", "Runtime version", runtimeVersion)

	s3Uri := strings.TrimPrefix(*sourceURI, "s3://")
	urlParts := strings.Split(s3Uri, "/")
	bucket := urlParts[0]
	path := strings.Join(urlParts[1:], "/")
	p.Log.Info("!-Bucket -------------!", "Bucket", bucket)
	p.Log.Info("!-Bucket -------------!", "Path", path)

	// serviceAccount := &v1.ServiceAccount{}
	// err := p.client.Get(context.TODO(), types.NamespacedName{Name: serviceAccountName,
	// 	Namespace: isvc.Namespace}, serviceAccount)

	// if err != nil {
	// 	p.Log.Error(err, "Failed to find service account", "ServiceAccountName", serviceAccountName,
	// 		"Namespace", isvc.Namespace)
	// 	return nil
	// }

	// for _, secretRef := range serviceAccount.Secrets {
	// 	p.Log.Info("!!!! found secret", "SecretName", secretRef.Name)
	// 	secret := &v1.Secret{}
	// 	err := p.client.Get(context.TODO(), types.NamespacedName{Name: secretRef.Name,
	// 		Namespace: isvc.Namespace}, secret)
	// 	if err != nil {
	// 		p.Log.Error(err, "Failed to find secret", "SecretName", secretRef.Name)
	// 		continue
	// 	}
	// }

	modelType := map[string]interface{}{
		"name": framework,
	}

	if *runtimeVersion != "" && runtimeVersion != nil {
		modelType["version"] = *runtimeVersion
	}

	predictor_obj := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "wmlserving.ai.ibm.com/v1",
			"kind":       "Predictor",
			"metadata": map[string]interface{}{
				"name":      isvc.Name,
				"namespace": isvc.Namespace,
			},
			"spec": map[string]interface{}{
				"modelType": modelType,
				"path":      path,
				"storage": map[string]interface{}{
					"s3": map[string]interface{}{
						"secretKey": serviceAccountName,
						"bucket":    bucket,
					},
				},
			},
		},
	}

	config, err := rest.InClusterConfig()
	if err != nil {
		p.Log.Info("Config not loaded")
	}

	client, err := dynamic.NewForConfig(config)
	if err != nil {
		p.Log.Info("Dynamic client not loaded")
	}

	gvr := schema.GroupVersionResource{
		Group:    "wmlserving.ai.ibm.com",
		Version:  "v1",
		Resource: "predictors",
	}
	result1 := client.Resource(gvr).Namespace(isvc.Namespace)
	_, err = result1.Get(context.TODO(), isvc.Name, metav1.GetOptions{})
	if err != nil {
		p.Log.Info("!--****** No predictor found. Creating... ******!")
		result, err := result1.Create(context.TODO(), predictor_obj, metav1.CreateOptions{})
		if err != nil {
			p.Log.Info("Error creating WML Predictor")
			return err
		}
		p.Log.Info("!--****** Created ******!", "Predictor", result.GetName())
	}

	fmt.Println("Going to return nil....")

	// p.Log.Info("!--****** Predictor Object ******!", predictor_obj)
	// isvc.Status.PropagateStatus(v1beta1.PredictorComponent, status)

	// isvc.Status.SetCondition()
	// statusSpec, ok := ss.Components[v1beta1.PredictorComponent]
	// if !ok {
	// 	ss.Components[component] = ComponentStatusSpec{}
	// }

	// isvc.Status.SetCondition(v1alpha1api.InferenceServiceReady, &apis.Condition{
	// 	Status: v1.ConditionTrue,
	// })

	return nil
}

func addLoggerAnnotations(logger *v1beta1.LoggerSpec, annotations map[string]string) bool {
	if logger != nil {
		annotations[constants.LoggerInternalAnnotationKey] = "true"
		if logger.URL != nil {
			annotations[constants.LoggerSinkUrlInternalAnnotationKey] = *logger.URL
		}
		annotations[constants.LoggerModeInternalAnnotationKey] = string(logger.Mode)
		return true
	}
	return false
}

func addLoggerContainerPort(container *v1.Container) {
	if container != nil {
		if container.Ports == nil || len(container.Ports) == 0 {
			port, _ := strconv.Atoi(constants.InferenceServiceDefaultAgentPort)
			container.Ports = []v1.ContainerPort{
				{
					ContainerPort: int32(port),
				},
			}
		}
	}
}

func addBatcherAnnotations(batcher *v1beta1.Batcher, annotations map[string]string) bool {
	if batcher != nil {
		annotations[constants.BatcherInternalAnnotationKey] = "true"

		if batcher.MaxBatchSize != nil {
			s := strconv.Itoa(*batcher.MaxBatchSize)
			annotations[constants.BatcherMaxBatchSizeInternalAnnotationKey] = s
		}
		if batcher.MaxLatency != nil {
			s := strconv.Itoa(*batcher.MaxLatency)
			annotations[constants.BatcherMaxLatencyInternalAnnotationKey] = s
		}
		if batcher.Timeout != nil {
			s := strconv.Itoa(*batcher.Timeout)
			annotations[constants.BatcherTimeoutInternalAnnotationKey] = s
		}
		return true
	}
	return false
}

func addBatcherContainerPort(container *v1.Container) {
	if container != nil {
		if container.Ports == nil || len(container.Ports) == 0 {
			port, _ := strconv.Atoi(constants.InferenceServiceDefaultAgentPort)
			container.Ports = []v1.ContainerPort{
				{
					ContainerPort: int32(port),
				},
			}
		}
	}
}

func addAgentAnnotations(isvc *v1beta1.InferenceService, annotations map[string]string, isvcConfig *v1beta1.InferenceServicesConfig) bool {
	if v1beta1utils.IsMMSPredictor(&isvc.Spec.Predictor, isvcConfig) {
		annotations[constants.AgentShouldInjectAnnotationKey] = "true"
		shardStrategy := memory.MemoryStrategy{}
		for _, id := range shardStrategy.GetShard(isvc) {
			multiModelConfigMapName := constants.ModelConfigName(isvc.Name, id)
			annotations[constants.AgentModelConfigVolumeNameAnnotationKey] = multiModelConfigMapName
			annotations[constants.AgentModelConfigMountPathAnnotationKey] = constants.ModelConfigDir
			annotations[constants.AgentModelDirAnnotationKey] = constants.ModelDir
		}
		return true
	}
	return false
}
