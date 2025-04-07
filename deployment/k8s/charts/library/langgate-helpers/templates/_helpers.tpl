{{/*
Define standard Kubernetes labels following the recommended label schema:
https://kubernetes.io/docs/concepts/overview/working-with-objects/common-labels/

These helpers use the standard app.kubernetes.io/* labels that are recognized
by many Kubernetes tools and make resources easier to query and manage.
*/}}

{{/*
Common selector labels - minimal set needed for pod selection by services/deployments
*/}}
{{- define "langgate-helpers.selectorLabels" -}}
app.kubernetes.io/name: {{ .name }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Common labels with additional metadata - used for all resources
Helm automatically adds app.kubernetes.io/managed-by and helm.sh/chart labels
*/}}
{{- define "langgate-helpers.labels" -}}
{{ include "langgate-helpers.selectorLabels" . }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/part-of: langgate
{{- end }}

{{/*
Get full image reference with registry - provides standardized image reference format
*/}}
{{- define "langgate-helpers.image" -}}
{{- $registry := .global.imageRegistry | default "" }}
{{- $repository := .image.repository }}
{{- $tag := .image.tag | default "latest" }}
{{- if $registry }}
{{- printf "%s/%s:%s" $registry $repository $tag }}
{{- else }}
{{- printf "%s:%s" $repository $tag }}
{{- end }}
{{- end }}

{{/*
Get namespace - simplifies namespace handling across templates
*/}}
{{- define "langgate-helpers.namespace" -}}
{{- .Values.global.namespace | default .Release.Namespace }}
{{- end }}

{{/*
Get configmap name - handles existing configmap reference or default
*/}}
{{- define "langgate-helpers.configMapName" -}}
{{- if hasKey .Values "config" }}
{{- .Values.config.existingConfigMap | default .Values.config.name }}
{{- else -}}
{{- "langgate-config" -}}
{{- end }}
{{- end }}

{{/*
Get secret name - handles existing secret reference or default
*/}}
{{- define "langgate-helpers.secretName" -}}
{{- if hasKey .Values "secrets" }}
{{- .Values.secrets.existingSecret | default .Values.secrets.name }}
{{- else -}}
{{- "langgate-secrets" -}}
{{- end }}
{{- end }}

{{/*
Standard volume mounts for configuration - provides consistent config volume mounting
*/}}
{{- define "langgate-helpers.configVolumeMounts" -}}
- name: config-volume
  mountPath: /etc/langgate
  readOnly: true
{{- end }}

{{/*
Standard volumes for configuration - provides consistent config volume definition
*/}}
{{- define "langgate-helpers.configVolumes" -}}
- name: config-volume
  configMap:
    name: {{ include "langgate-helpers.configMapName" . }}
{{- end }}

{{/*
Get processor service name - handles parent-child chart references
*/}}
{{- define "langgate-helpers.processorServiceName" -}}
{{- $root := . -}}
{{- if hasKey $root.Values "langgate_processor" }}
{{- $root.Values.langgate_processor.name }}
{{- else -}}
{{- "langgate-processor" -}}
{{- end }}
{{- end }}

{{/*
Get server service name - handles parent-child chart references
*/}}
{{- define "langgate-helpers.serverServiceName" -}}
{{- $root := . -}}
{{- if hasKey $root.Values "langgate_server" }}
{{- $root.Values.langgate_server.name }}
{{- else -}}
{{- "langgate-server" -}}
{{- end }}
{{- end }}
