(function () {
  "use strict";
  // Smart-Model-Swap dashboard tab. Plain IIFE against the host SDK (no build step).
  // SmartModelProvider: engines (llama.cpp managed; Ollama / vLLM / LM Studio under
  // construction) + a full launch-parameter editor modelled on Llamacpp-Model-Launcher.
  const SDK = window.__HERMES_PLUGIN_SDK__;
  if (!SDK || !window.__HERMES_PLUGINS__) return;

  const React = SDK.React;
  const h = React.createElement;
  const { useState, useEffect, useCallback, useRef } = SDK.hooks;
  const C = SDK.components || {};
  const API = "/api/plugins/smart-model-swap";
  const BUSY = ["grace", "swapping", "restoring", "starting"];

  // ------------------------------------------------------------------ i18n
  // UI strings only. Data coming from the backend (errors, warnings, notes,
  // catalog groups) is localized by the backend itself via ?lang=.
  const I18N = {
    es: {
      // engines
      engineSaved: (label) => label + " guardado.",
      failed: "falló",
      activeEngineMsg: (label) => "Motor activo: " + label,
      underConstruction: "🚧 En construcción",
      underConstructionShort: "en construcción",
      activeTag: "activo",
      activate: "activar",
      portLink: "puerto",
      configure: "configurar",
      online: "en línea",
      modelCount: (n) => n + " modelo(s)",
      noResponse: "sin respuesta",
      binaryNotFound: "Binario no encontrado: configura la ruta del motor ya descargado.",
      binaryPathLabel: "Binario (ruta a llama-server o a su carpeta)",
      test: "Probar",
      hostLabel: "Host",
      portLabel: "Puerto",
      envVarsLines: "Variables de entorno (KEY=valor, una por línea)",
      commonArgs: "Args comunes (todos los modelos de este motor)",
      save: "Guardar",
      cancel: "Cancelar",
      engineApplyNote: "Host, puerto y -m los pone SMP; se aplican en el próximo arranque.",
      enginesNote: "SmartModelProvider arranca el motor elegido por su cuenta (sin systemd). Solo necesita el motor ya descargado y su ruta.",
      // parameter editor / catalog
      switchNoValue: "switch (sin valor)",
      choose: "— elegir —",
      removedInBuild: "eliminado en este build",
      notInBuild: "no existe en este build",
      alreadyAdded: "ya añadido",
      add: "Añadir",
      searchParamPlaceholder: "Buscar parámetro (nombre, flag o descripción)…",
      catalogChecked: (v) => "Catálogo comprobado contra tu build: " + v,
      noResults: "Sin resultados.",
      // model editor
      couldNotLoad: "no se pudo cargar",
      loading: "Cargando…",
      paramMustStartWithDash: "El parámetro debe empezar por '-' o '--'.",
      confirmAddAgain: (key) => "'" + key + "' ya está en la lista. ¿Añadirlo otra vez?",
      backendRejected: "el backend rechazó la acción",
      savedWithWarnings: (w) => "Guardado. " + w,
      modelSaved: (name) => "Modelo '" + name + "' guardado.",
      confirmImportReplace: "Se reemplazarán GGUF, mmproj, entorno y parámetros actuales. ¿Continuar?",
      importedLine: (n) => "Línea importada: " + n + " parámetros.",
      duplicatePrompt: (name) => "Nombre de la copia de '" + name + "':",
      copySuffix: "-copia",
      confirmDeleteModel: (name) => "¿Eliminar '" + name + "' del registro?",
      nameLabel: "Nombre",
      namePlaceholder: "p. ej. Qwen3.5-9B_Q5",
      nameRules: "Letras (mayúsculas/minúsculas), dígitos y . _ - + ( ) [ ] , — sin espacios, / ni :",
      engineLabel: "Motor",
      roleLabel: "Rol",
      rolePlaceholder: "imágenes, nano…",
      minVramLabel: "VRAM mínima (GB)",
      loadTimeoutLabel: "Timeout de carga (s)",
      ggufLabel: "Modelo (GGUF)",
      ggufPlaceholder: "/ruta/al/modelo.gguf",
      mmprojLabel: "mmproj (visión, opcional)",
      mmprojPlaceholder: "/ruta/al/mmproj.gguf",
      modelBinaryLabel: "Binario del motor para este modelo",
      modelBinaryPlaceholder: "vacío = el del motor (p. ej. build CUDA vs Vulkan)",
      envVarsLabel: "Variables de entorno (KEY=valor)",
      descriptionLabel: "Descripción",
      launchParams: (n) => "Parámetros de arranque (" + n + ")",
      closeImport: "cerrar importación",
      importCmdline: "importar línea de comando",
      importPlaceholder: "llama-server -m /ruta/modelo.gguf -c 32768 -ngl 99 --temp 0.7 …",
      importBtn: "Importar",
      noOwnParams: "Sin parámetros propios: solo se usarán los args comunes del motor.",
      ctxSharedHint: "El contexto (-c / --ctx-size) es compartido: al guardar se aplica a todos los modelos del motor, porque Hermes usa un único contexto para el modelo activo.",
      ctxWillApply: (ctx, names) => "Al guardar, el contexto " + ctx + " se aplicará también a: " + names + ".",
      ctxWillInherit: (ctx) => "Sin -c: al guardar se usará el contexto compartido (" + ctx + ").",
      ctxSharedBadge: "compartido",
      moveUp: "subir",
      moveDown: "bajar",
      removeParam: (key) => "quitar " + key,
      paramPlaceholder: "--parámetro",
      valuePlaceholder: "valor (vacío = switch)",
      previewValidate: "Vista previa y validar",
      reset: "Restablecer",
      duplicate: "Duplicar",
      deleteLink: "eliminar",
      confirmCloseUnsaved: "Hay cambios sin guardar. ¿Cerrar el editor?",
      close: "cerrar",
      binaryAccepts: (build) => "El binario acepta la línea de comandos" + (build ? " (" + build + ")" : "") + ".",
      rejected: (err) => "Rechazada: " + err,
      previewOfSaved: "(vista previa de la versión guardada o de la última validación)",
      unrecognized: (list) => "No reconocidos por este build: " + list,
      suggestedFixes: (s) => "Correcciones sugeridas: " + s,
      fixesApplied: "Correcciones aplicadas (sin guardar).",
      applyFixes: "Aplicar correcciones",
      // sampling params
      paramTemperature: "Temperatura",
      paramTopP: "Top P",
      paramTopK: "Top K",
      paramMinP: "Min P",
      paramPresencePenalty: "Presence penalty",
      paramRepeatPenalty: "Repeat penalty",
      notANumber: "no es un número",
      mustBeInteger: "debe ser entero",
      outOfRange: (min, max) => "fuera de " + min + "–" + max,
      noModels: "No hay modelos en el registro.",
      modelLabel: "Modelo",
      loadedSuffix: "cargado",
      activePresetLabel: "Preset activo",
      presetActiveMsg: (p, model) => "Preset '" + p + "' activo: se aplicará la próxima vez que se cargue " + model + ".",
      noPresetMsg: (model) => "Sin preset: " + model + " cargará con sus parámetros base.",
      basePreset: "Base (sin preset)",
      presetUpdated: (p) => "Preset '" + p + "' actualizado.",
      baseParamsSaved: (model) => "Parámetros base de " + model + " guardados.",
      saveToPreset: (p) => "Guardar en '" + p + "'",
      saveBaseParams: "Guardar parámetros base",
      discard: "Descartar",
      confirmDeletePreset: (p, model) => "¿Eliminar el preset '" + p + "' de " + model + "?",
      presetDeleted: (p) => "Preset '" + p + "' eliminado.",
      deletePreset: "eliminar preset",
      newPresetPlaceholder: "nombre del nuevo preset",
      confirmOverwritePreset: (p) => "El preset '" + p + "' ya existe. ¿Sobrescribirlo?",
      presetSavedActive: (p) => "Preset '" + p + "' guardado y activo.",
      saveAsPreset: "Guardar como preset",
      paramsApplyNote: (isLoaded) => "Se aplican la próxima vez que se cargue el modelo" + (isLoaded ? "; para aplicarlos al que está cargado hay que recargarlo." : "."),
      // provider
      installed: "Instalado",
      installedOutdated: "Instalado · desactualizado",
      folderTaken: "Carpeta ocupada",
      notInstalled: "No instalado",
      foreignFolder: (path) => "Ya existe " + path + " y no lo creó este plugin; no se toca. Quítalo o renómbralo para poder instalar.",
      providerExplain: "Expone los modelos del registro como un proveedor de Hermes, para elegirlos en /model con su contexto real y sin configurar custom_providers.",
      providerInstalled: "SmartModelProvider instalado.",
      installProvider: "Instalar SmartModelProvider",
      createsFolder: (path) => "Crea la carpeta " + path + ". No funciona sin este plugin y se quita desde aquí.",
      tokens: (n) => n + " tokens",
      providerHowTo: "Elígelo en Hermes con /model → SmartModelProvider. Elegir ahí un modelo no lo carga: el cambio sigue siendo el interruptor. La lista llega a Hermes al reiniciar gateway o serve.",
      providerUpdated: "SmartModelProvider actualizado.",
      update: "Actualizar",
      confirmRemoveProvider: "¿Quitar SmartModelProvider? Hermes dejará de ofrecer estos modelos como proveedor.",
      providerRemoved: "SmartModelProvider quitado.",
      removeLink: "quitar",
      // page
      couldNotLoadState: (e) => "No se pudo cargar el estado: " + e,
      runnerDied: "El runner murió a mitad del swap.",
      lastSwapFailed: "El último swap falló.",
      cutoverIn: (s, target) => "corte en " + s + " s → " + target,
      switchingToPrimary: (m) => "Cambiando al primario (" + m + ")…",
      switchingToSecondary: (m) => "Cambiando al secundario (" + m + ")…",
      slotsExchanged: "Primario y secundario intercambiados.",
      primarySet: (m) => "Primario: " + m,
      secondarySet: (m) => "Secundario: " + m,
      dryRunOk: (m, gbAvail) => "Dry-run OK: " + m + " cabe (" + gbAvail + " GB disponibles tras descargar el actual).",
      dryRunFail: (detail) => "Dry-run: " + detail,
      swapInProgress: "hay un swap en curso",
      noBackend: (e) => "Sin conexión con el backend: " + e,
      activeModelTitle: "Modelo activo",
      unregisteredModel: "modelo no registrado",
      startingUp: "arrancando…",
      gpuJobUnloaded: "LLM descargado por un trabajo de GPU",
      gpuJobUnloadedNote: (job, min) => "Trabajo en curso" + (job ? " (" + job + ")" : "") + ", " + min + " min. Al terminar vuelve un modelo solo; no hagas swap mientras tanto.",
      gpuJobRunningNote: (job, min) => "Trabajo de GPU en curso" + (job ? " (" + job + ")" : "") + ", " + min + " min. Al terminar sigue respondiendo el modelo cargado.",
      gpuModeTitle: "Trabajos de GPU · modo",
      gpuModeNote: "Qué hacer con el modelo cuando llega un trabajo de alta carga en VRAM (ComfyUI, vídeo, 3D…). Lo aplica gpu_job.sh auto.",
      gpuModeAria: "Modo para trabajos de GPU",
      gpuModeSaved: (label) => "Modo para trabajos de GPU: " + label,
      gpuModeNow: "En este equipo, ahora:",
      settingsTitle: "Automatización",
      setAfterJob: "Tras un trabajo de GPU responde",
      setAfterJobHint: "El pequeño responde enseguida; el primario tarda en recargar.",
      setAfterSmall: "Modelo pequeño",
      setAfterPrimary: "Primario",
      setIdle: "Volver al primario tras inactividad",
      setIdleHint: "Minutos sin actividad en el modelo pequeño. 0 lo desactiva. Se aplica desde el próximo cambio de modelo.",
      setIdleUnit: "min",
      setRecover: "Recuperación automática",
      setRecoverHint: "Si un modelo falla al cargar o se cae, se carga otro que quepa.",
      setAutostart: "Cargar el primario al iniciar el gateway",
      setAutostartHint: "Al arrancar Hermes siempre se carga el modelo primario, aunque haya quedado otro sirviendo.",
      setSaved: "Ajuste guardado.",
      setIdleBad: "Escribe un número entero entre 0 y 720.",
      on: "activado", off: "desactivado",
      serverDown: "servidor caído",
      stateText: (s) => "estado: " + s,
      engineText: (s) => "motor: " + s,
      serverText: (s) => "servidor: " + s,
      primary: "Primario",
      secondary: "Secundario",
      toggleAria: "Cambiar entre el modelo primario y el secundario",
      backToPrimary: "Volver al primario",
      switchToSecondary: "Cambiar al secundario",
      loadPrimary: "Cargar el primario",
      testSwap: "Probar swap",
      selectorNote: "Cambiar un selector no carga nada: el interruptor hace el swap.",
      engineWipNote: "El motor activo está en construcción: el swap solo funciona con llama.cpp.",
      vramTitle: "VRAM",
      noNvidiaSmi: "nvidia-smi no disponible.",
      freeText: (v) => "libre: " + v,
      comfyUp: "activo",
      comfyDown: "apagado",
      enginesTitle: "Motores · SmartModelProvider",
      modelsTitle: (n) => "Modelos (" + n + ")",
      ggufFound: "GGUF encontrado",
      ggufNotFound: "GGUF no encontrado",
      presetTag: (p) => "preset: " + p,
      primaryTag: "primario",
      secondaryTag: "secundario",
      edit: "editar",
      newModel: "Nuevo modelo",
      editorNote: "El editor permite todos los flags de arranque, entorno, binario por modelo e importar una línea de comando.",
      editorTitle: (name) => "Editor · " + name,
      samplingTitle: "Muestreo y presets",
      providerTitle: "Proveedor en Hermes (/model)",
      runnerLogTitle: "Log del runner",
      noLogsYet: "(sin logs todavía)"
    },
    en: {
      // engines
      engineSaved: (label) => label + " saved.",
      failed: "failed",
      activeEngineMsg: (label) => "Active engine: " + label,
      underConstruction: "🚧 Under construction",
      underConstructionShort: "under construction",
      activeTag: "active",
      activate: "activate",
      portLink: "port",
      configure: "configure",
      online: "online",
      modelCount: (n) => n + " model(s)",
      noResponse: "no response",
      binaryNotFound: "Binary not found: set the path to the already-downloaded engine.",
      binaryPathLabel: "Binary (path to llama-server or its folder)",
      test: "Test",
      hostLabel: "Host",
      portLabel: "Port",
      envVarsLines: "Environment variables (KEY=value, one per line)",
      commonArgs: "Common args (all models of this engine)",
      save: "Save",
      cancel: "Cancel",
      engineApplyNote: "SMP sets host, port and -m; changes apply on the next start.",
      enginesNote: "SmartModelProvider starts the chosen engine on its own (no systemd). It only needs the engine already downloaded and its path.",
      // parameter editor / catalog
      switchNoValue: "switch (no value)",
      choose: "— choose —",
      removedInBuild: "removed in this build",
      notInBuild: "not in this build",
      alreadyAdded: "already added",
      add: "Add",
      searchParamPlaceholder: "Search parameter (name, flag or description)…",
      catalogChecked: (v) => "Catalog checked against your build: " + v,
      noResults: "No results.",
      // model editor
      couldNotLoad: "could not load",
      loading: "Loading…",
      paramMustStartWithDash: "The parameter must start with '-' or '--'.",
      confirmAddAgain: (key) => "'" + key + "' is already in the list. Add it again?",
      backendRejected: "the backend rejected the action",
      savedWithWarnings: (w) => "Saved. " + w,
      modelSaved: (name) => "Model '" + name + "' saved.",
      confirmImportReplace: "This will replace the current GGUF, mmproj, environment and parameters. Continue?",
      importedLine: (n) => "Command line imported: " + n + " parameters.",
      duplicatePrompt: (name) => "Name for the copy of '" + name + "':",
      copySuffix: "-copy",
      confirmDeleteModel: (name) => "Remove '" + name + "' from the registry?",
      nameLabel: "Name",
      namePlaceholder: "e.g. Qwen3.5-9B_Q5",
      nameRules: "Letters (upper/lower case), digits and . _ - + ( ) [ ] , — no spaces, / or :",
      engineLabel: "Engine",
      roleLabel: "Role",
      rolePlaceholder: "images, nano…",
      minVramLabel: "Min. VRAM (GB)",
      loadTimeoutLabel: "Load timeout (s)",
      ggufLabel: "Model (GGUF)",
      ggufPlaceholder: "/path/to/model.gguf",
      mmprojLabel: "mmproj (vision, optional)",
      mmprojPlaceholder: "/path/to/mmproj.gguf",
      modelBinaryLabel: "Engine binary for this model",
      modelBinaryPlaceholder: "empty = the engine's (e.g. CUDA vs Vulkan build)",
      envVarsLabel: "Environment variables (KEY=value)",
      descriptionLabel: "Description",
      launchParams: (n) => "Launch parameters (" + n + ")",
      closeImport: "close import",
      importCmdline: "import command line",
      importPlaceholder: "llama-server -m /path/model.gguf -c 32768 -ngl 99 --temp 0.7 …",
      importBtn: "Import",
      noOwnParams: "No model-specific parameters: only the engine's common args will be used.",
      ctxSharedHint: "The context (-c / --ctx-size) is shared: saving applies it to every model of the engine, because Hermes uses a single context for the active model.",
      ctxWillApply: (ctx, names) => "On save, context " + ctx + " will also be applied to: " + names + ".",
      ctxWillInherit: (ctx) => "No -c: on save the shared context (" + ctx + ") will be used.",
      ctxSharedBadge: "shared",
      moveUp: "move up",
      moveDown: "move down",
      removeParam: (key) => "remove " + key,
      paramPlaceholder: "--parameter",
      valuePlaceholder: "value (empty = switch)",
      previewValidate: "Preview & validate",
      reset: "Reset",
      duplicate: "Duplicate",
      deleteLink: "delete",
      confirmCloseUnsaved: "There are unsaved changes. Close the editor?",
      close: "close",
      binaryAccepts: (build) => "The binary accepts the command line" + (build ? " (" + build + ")" : "") + ".",
      rejected: (err) => "Rejected: " + err,
      previewOfSaved: "(preview of the saved version or of the last validation)",
      unrecognized: (list) => "Not recognized by this build: " + list,
      suggestedFixes: (s) => "Suggested fixes: " + s,
      fixesApplied: "Fixes applied (not saved).",
      applyFixes: "Apply fixes",
      // sampling params
      paramTemperature: "Temperature",
      paramTopP: "Top P",
      paramTopK: "Top K",
      paramMinP: "Min P",
      paramPresencePenalty: "Presence penalty",
      paramRepeatPenalty: "Repeat penalty",
      notANumber: "not a number",
      mustBeInteger: "must be an integer",
      outOfRange: (min, max) => "out of range " + min + "–" + max,
      noModels: "No models in the registry.",
      modelLabel: "Model",
      loadedSuffix: "loaded",
      activePresetLabel: "Active preset",
      presetActiveMsg: (p, model) => "Preset '" + p + "' active: it will apply the next time " + model + " is loaded.",
      noPresetMsg: (model) => "No preset: " + model + " will load with its base parameters.",
      basePreset: "Base (no preset)",
      presetUpdated: (p) => "Preset '" + p + "' updated.",
      baseParamsSaved: (model) => "Base parameters for " + model + " saved.",
      saveToPreset: (p) => "Save to '" + p + "'",
      saveBaseParams: "Save base parameters",
      discard: "Discard",
      confirmDeletePreset: (p, model) => "Delete preset '" + p + "' from " + model + "?",
      presetDeleted: (p) => "Preset '" + p + "' deleted.",
      deletePreset: "delete preset",
      newPresetPlaceholder: "new preset name",
      confirmOverwritePreset: (p) => "Preset '" + p + "' already exists. Overwrite it?",
      presetSavedActive: (p) => "Preset '" + p + "' saved and active.",
      saveAsPreset: "Save as preset",
      paramsApplyNote: (isLoaded) => "They apply the next time the model is loaded" + (isLoaded ? "; to apply them to the loaded model, reload it." : "."),
      // provider
      installed: "Installed",
      installedOutdated: "Installed · outdated",
      folderTaken: "Folder in use",
      notInstalled: "Not installed",
      foreignFolder: (path) => path + " already exists and was not created by this plugin; it is left untouched. Remove or rename it to install.",
      providerExplain: "Exposes the registry's models as a Hermes provider, so you can pick them in /model with their real context and without configuring custom_providers.",
      providerInstalled: "SmartModelProvider installed.",
      installProvider: "Install SmartModelProvider",
      createsFolder: (path) => "Creates the folder " + path + ". It does not work without this plugin and is removed from here.",
      tokens: (n) => n + " tokens",
      providerHowTo: "Pick it in Hermes with /model → SmartModelProvider. Choosing a model there does not load it: the switch still does the swap. The list reaches Hermes when gateway or serve restarts.",
      providerUpdated: "SmartModelProvider updated.",
      update: "Update",
      confirmRemoveProvider: "Remove SmartModelProvider? Hermes will stop offering these models as a provider.",
      providerRemoved: "SmartModelProvider removed.",
      removeLink: "remove",
      // page
      couldNotLoadState: (e) => "Could not load the state: " + e,
      runnerDied: "The runner died mid-swap.",
      lastSwapFailed: "The last swap failed.",
      cutoverIn: (s, target) => "cutover in " + s + " s → " + target,
      switchingToPrimary: (m) => "Switching to primary (" + m + ")…",
      switchingToSecondary: (m) => "Switching to secondary (" + m + ")…",
      slotsExchanged: "Primary and secondary swapped.",
      primarySet: (m) => "Primary: " + m,
      secondarySet: (m) => "Secondary: " + m,
      dryRunOk: (m, gbAvail) => "Dry-run OK: " + m + " fits (" + gbAvail + " GB available after unloading the current one).",
      dryRunFail: (detail) => "Dry-run: " + detail,
      swapInProgress: "a swap is in progress",
      noBackend: (e) => "No connection to the backend: " + e,
      activeModelTitle: "Active model",
      unregisteredModel: "unregistered model",
      startingUp: "starting…",
      gpuJobUnloaded: "LLM unloaded for a GPU job",
      gpuJobUnloadedNote: (job, min) => "Job running" + (job ? " (" + job + ")" : "") + ", " + min + " min. A model comes back by itself when it ends; do not swap meanwhile.",
      gpuJobRunningNote: (job, min) => "GPU job running" + (job ? " (" + job + ")" : "") + ", " + min + " min. The loaded model keeps answering when it ends.",
      gpuModeTitle: "GPU jobs · mode",
      gpuModeNote: "What to do with the model when a VRAM-heavy job arrives (ComfyUI, video, 3D…). Applied by gpu_job.sh auto.",
      gpuModeAria: "GPU job mode",
      gpuModeSaved: (label) => "GPU job mode: " + label,
      gpuModeNow: "On this machine, now:",
      settingsTitle: "Automation",
      setAfterJob: "After a GPU job, answer with",
      setAfterJobHint: "The small one answers right away; the primary takes time to reload.",
      setAfterSmall: "Small model",
      setAfterPrimary: "Primary",
      setIdle: "Back to the primary when idle",
      setIdleHint: "Idle minutes on the small model. 0 disables it. Applies from the next model change.",
      setIdleUnit: "min",
      setRecover: "Auto-recover",
      setRecoverHint: "If a model fails to load or dies, another one that fits is loaded.",
      setAutostart: "Load the primary when the gateway starts",
      setAutostartHint: "When Hermes starts the primary model is always loaded, even if another one was left serving.",
      setSaved: "Setting saved.",
      setIdleBad: "Enter a whole number between 0 and 720.",
      on: "on", off: "off",
      serverDown: "server down",
      stateText: (s) => "state: " + s,
      engineText: (s) => "engine: " + s,
      serverText: (s) => "server: " + s,
      primary: "Primary",
      secondary: "Secondary",
      toggleAria: "Switch between the primary and the secondary model",
      backToPrimary: "Back to primary",
      switchToSecondary: "Switch to secondary",
      loadPrimary: "Load the primary",
      testSwap: "Test swap",
      selectorNote: "Changing a selector loads nothing: the switch does the swap.",
      engineWipNote: "The active engine is under construction: swap only works with llama.cpp.",
      vramTitle: "VRAM",
      noNvidiaSmi: "nvidia-smi not available.",
      freeText: (v) => "free: " + v,
      comfyUp: "running",
      comfyDown: "off",
      enginesTitle: "Engines · SmartModelProvider",
      modelsTitle: (n) => "Models (" + n + ")",
      ggufFound: "GGUF found",
      ggufNotFound: "GGUF not found",
      presetTag: (p) => "preset: " + p,
      primaryTag: "primary",
      secondaryTag: "secondary",
      edit: "edit",
      newModel: "New model",
      editorNote: "The editor supports every launch flag, environment, per-model binary and importing a command line.",
      editorTitle: (name) => "Editor · " + name,
      samplingTitle: "Sampling & presets",
      providerTitle: "Hermes provider (/model)",
      runnerLogTitle: "Runner log",
      noLogsYet: "(no logs yet)"
    }
  };
  let LANG = 'es';
  function langOf(loc) { return /^es([-_]|$)/i.test(String(loc || '')) ? 'es' : 'en'; }
  function t(key, ...args) { const d = I18N[LANG] || I18N.es; let v = d[key]; if (v === undefined) v = I18N.es[key]; if (v === undefined) return key; return typeof v === 'function' ? v(...args) : v; }

  // Host primitives when present, plain elements otherwise (older hosts).
  function Card(props) {
    if (!C.Card) return h("section", { className: "sms-item sms-col", style: { alignItems: "stretch" } }, h("strong", null, props.title), props.children);
    return h(C.Card, null,
      h(C.CardHeader, null, h(C.CardTitle, null, props.title)),
      h(C.CardContent, null, h("div", { className: "sms-col" }, props.children)));
  }
  function Btn(props) {
    const p = { onClick: props.onClick, disabled: !!props.disabled, title: props.title };
    if (C.Button) return h(C.Button, Object.assign({ size: "sm", variant: props.variant }, p), props.children);
    return h("button", Object.assign({ type: "button", className: "sms-input" }, p), props.children);
  }
  function Tag(props) {
    return C.Badge ? h(C.Badge, { variant: "outline" }, props.children) : h("span", { className: "sms-muted" }, props.children);
  }
  function Msg(props) {
    const m = props.msg;
    if (!m) return null;
    return h("div", { className: m.kind === "bad" ? "sms-error" : m.kind === "warn" ? "sms-warn" : "sms-ok" }, m.text);
  }

  // Every backend request carries the UI language so the backend can localize
  // its own messages (errors, warnings, notes, catalog groups).
  function withLang(path) { return path + (path.indexOf('?') >= 0 ? '&' : '?') + 'lang=' + LANG; }
  function get(path) { return SDK.fetchJSON(API + withLang(path)); }
  function post(path, body) {
    return SDK.fetchJSON(API + withLang(path), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {})
    });
  }
  function errText(e) { return String((e && e.message) || e); }
  function gb(mib) { return (mib / 1024).toFixed(1) + " GB"; }
  function tone(state) {
    if (BUSY.indexOf(state) >= 0) return "warn";
    if (state === "failed") return "bad";
    return "good";
  }

  // ------------------------------------------------------------ command-line helpers
  // Same quoting the backend tokenizer (smp/cmdline.py) understands.
  function quote(s) {
    s = String(s);
    if (s !== "" && /^[A-Za-z0-9_\-.,:\/=+@%]+$/.test(s)) return s;
    return '"' + s.replace(/"/g, '\\"') + '"';
  }
  function joinArgs(list) { return (list || []).map(quote).join(" "); }
  function envToText(env) {
    return Object.keys(env || {}).map(function (k) { return k + "=" + env[k]; }).join("\n");
  }
  let rowSeq = 0;
  function mkRow(key, value) { rowSeq += 1; return { id: "r" + rowSeq, key: key, value: value }; }
  function withIds(rows) { return (rows || []).map(function (r) { return mkRow(r.key, r.value == null ? null : String(r.value)); }); }
  function plainRows(rows) { return rows.map(function (r) { return { key: r.key.trim(), value: r.value == null ? null : String(r.value) }; }); }

  // ------------------------------------------------------------------ engines card
  function EngineCard(props) {
    const e = props.engine, health = props.health || {};
    const wip = e.status === "wip";
    const [open, setOpen] = useState(false);
    const [f, setF] = useState(null);
    const [pending, setPending] = useState(false);
    const [msg, setMsg] = useState(null);
    const [version, setVersion] = useState(null);

    function start() {
      setF({ bin: e.bin || "", host: e.host || "127.0.0.1", port: String(e.port || ""),
        env: envToText(e.env), common: joinArgs(e.common_args) });
      setOpen(true); setMsg(null);
    }
    function set(k) { return function (ev) { const v = ev.target.value; setF(function (o) { const n = Object.assign({}, o); n[k] = v; return n; }); }; }
    function save() {
      const body = { engine: e.id, host: f.host.trim(), port: parseInt(f.port, 10) || null };
      if (!wip) { body.bin = f.bin.trim(); body.env = f.env; body.common_args = f.common; }
      setPending(true); setMsg(null);
      post("/engines/update", body).then(function (r) {
        if (r && r.ok === false) setMsg({ kind: "bad", text: r.error });
        else {
          const w = (r && r.warnings) || [];
          setMsg(w.length ? { kind: "warn", text: w.join(" · ") } : { kind: "ok", text: t("engineSaved", e.label) });
          setOpen(false);
        }
      }).catch(function (x) { setMsg({ kind: "bad", text: errText(x) }); })
        .then(function () { setPending(false); props.reload(); });
    }
    function test() {
      setPending(true); setVersion(null);
      post("/engines/test", { engine: e.id, bin: f ? f.bin.trim() : "" }).then(function (r) {
        setVersion(r && r.ok ? { ok: true, text: r.version } : { ok: false, text: (r && r.error) || t("failed") });
      }).catch(function (x) { setVersion({ ok: false, text: errText(x) }); })
        .then(function () { setPending(false); });
    }
    function activate() {
      setPending(true); setMsg(null);
      post("/engines/select", { engine: e.id }).then(function (r) {
        setMsg(r && r.ok === false ? { kind: "bad", text: r.error } : { kind: "ok", text: t("activeEngineMsg", e.label) });
      }).catch(function (x) { setMsg({ kind: "bad", text: errText(x) }); })
        .then(function () { setPending(false); props.reload(); });
    }

    const n = (health.models && health.models.length) || 0;
    return h("div", { className: "sms-engine" + (e.active ? " active" : "") + (wip ? " wip" : "") },
      h("div", { className: "sms-row" },
        h("span", { className: "sms-dot " + (health.up ? "good" : wip ? "" : "bad") }),
        h("strong", { className: "grow" }, e.label),
        wip ? h("span", { className: "sms-wip" }, t("underConstruction")) : null,
        e.active ? h(Tag, null, t("activeTag")) : (!wip ? h("button", { className: "sms-link", disabled: pending, onClick: activate }, t("activate")) : null),
        !open ? h("button", { className: "sms-link", onClick: start }, wip ? t("portLink") : t("configure")) : null),
      h("div", { className: "sms-muted" },
        e.host + ":" + e.port + (health.up ? " · " + t("online") + (n ? " · " + t("modelCount", n) : "") : " · " + t("noResponse"))
        + (!wip && e.bin ? " · " + e.bin : "")),
      !wip && e.bin_ok === false ? h("div", { className: "sms-error" }, t("binaryNotFound")) : null,
      wip ? h("div", { className: "sms-muted" }, e.note) : null,
      open && f ? h("div", { className: "sms-col" },
        !wip ? h("label", { className: "sms-field" }, h("span", { className: "sms-cap" }, t("binaryPathLabel")),
          h("div", { className: "sms-row" },
            h("input", { className: "sms-input mono", style: { flex: 1 }, value: f.bin, onChange: set("bin") }),
            h(Btn, { variant: "outline", disabled: pending, onClick: test }, t("test")))) : null,
        version ? h("div", { className: version.ok ? "sms-ok" : "sms-error" }, version.text) : null,
        h("div", { className: "sms-form" },
          h("label", { className: "sms-field" }, h("span", { className: "sms-cap" }, t("hostLabel")),
            h("input", { className: "sms-input", value: f.host, onChange: set("host") })),
          h("label", { className: "sms-field" }, h("span", { className: "sms-cap" }, t("portLabel")),
            h("input", { className: "sms-input", value: f.port, inputMode: "numeric", onChange: function (ev) { const v = ev.target.value.replace(/[^0-9]/g, ""); setF(function (o) { return Object.assign({}, o, { port: v }); }); } }))),
        !wip ? h("label", { className: "sms-field" }, h("span", { className: "sms-cap" }, t("envVarsLines")),
          h("textarea", { className: "sms-input mono", rows: 2, value: f.env, onChange: set("env"), placeholder: "CUDA_VISIBLE_DEVICES=0,1" })) : null,
        !wip ? h("label", { className: "sms-field" }, h("span", { className: "sms-cap" }, t("commonArgs")),
          h("textarea", { className: "sms-input mono", rows: 2, value: f.common, onChange: set("common"), placeholder: "-ngl 99 --jinja --flash-attn on --metrics" })) : null,
        h("div", { className: "sms-row" },
          h(Btn, { disabled: pending, onClick: save }, t("save")),
          h(Btn, { variant: "outline", disabled: pending, onClick: function () { setOpen(false); setVersion(null); } }, t("cancel")),
          !wip ? h("span", { className: "sms-muted" }, t("engineApplyNote")) : null)) : null,
      h(Msg, { msg: msg }));
  }

  function Engines(props) {
    return h("div", { className: "sms-col" },
      (props.engines || []).map(function (e) {
        return h(EngineCard, { key: e.id, engine: e, health: (props.health || {})[e.id], reload: props.reload });
      }),
      h("div", { className: "sms-muted" }, t("enginesNote")));
  }

  // ------------------------------------------------------------- model editor
  const EMPTY = { name: "", original_name: null, engine: "llamacpp", gguf: "", mmproj: "", bin: "", env: "",
    role: "", description: "", min_free_vram_gb: "8", load_timeout_s: "240" };

  function draftFrom(m) {
    return {
      name: m.name, original_name: m.name, engine: m.engine || "llamacpp", gguf: m.gguf || "", mmproj: m.mmproj || "",
      bin: m.bin || "", env: envToText(m.env), role: m.role || "", description: m.description || "",
      min_free_vram_gb: String(m.min_free_vram_gb == null ? "" : m.min_free_vram_gb),
      load_timeout_s: String(m.load_timeout_s == null ? "" : m.load_timeout_s)
    };
  }
  function bodyOf(d, rows) {
    return {
      name: d.name.trim(), original_name: d.original_name, engine: d.engine, gguf: d.gguf.trim(), mmproj: d.mmproj.trim(),
      bin: d.bin.trim(), env: d.env, rows: plainRows(rows), role: d.role.trim(), description: d.description.trim(),
      min_free_vram_gb: parseFloat(String(d.min_free_vram_gb).replace(",", ".")) || 0,
      load_timeout_s: parseInt(d.load_timeout_s, 10) || 240
    };
  }
  function sig(d, rows) { return JSON.stringify(bodyOf(d, rows)); }

  function ParamValue(props) {
    const r = props.row, info = props.info;
    if (r.value == null) {
      return h("label", { className: "sms-row sms-muted", style: { flex: 1 } },
        h("input", { type: "checkbox", checked: true, onChange: props.onRemove }), t("switchNoValue"));
    }
    if (info && info.type === "select" && (info.options || []).length) {
      const opts = info.options.indexOf(r.value) >= 0 || r.value === "" ? info.options : [r.value].concat(info.options);
      return h("select", { className: "sms-select", style: { flex: 1 }, value: r.value, onChange: function (e) { props.onChange(e.target.value); } },
        r.value === "" ? h("option", { value: "" }, t("choose")) : null,
        opts.map(function (o) { return h("option", { key: o, value: o }, o); }));
    }
    return h("input", { className: "sms-input mono", style: { flex: 1 }, value: r.value, onChange: function (e) { props.onChange(e.target.value); } });
  }

  function Catalog(props) {
    const [q, setQ] = useState("");
    const groups = props.catalog ? props.catalog.groups || [] : [];
    const needle = q.trim().toLowerCase();
    const have = props.have || {};
    let shown = 0;
    const out = [];
    groups.forEach(function (g) {
      const ps = (g.parameters || []).filter(function (p) {
        if (!needle) return false;
        return (p.prefix + " " + p.name + " " + (p.description || "")).toLowerCase().indexOf(needle) >= 0;
      });
      if (!ps.length || shown >= 60) return;
      out.push(h("div", { key: g.name, className: "sms-cap sms-cat-group" }, g.name));
      ps.slice(0, 60 - shown).forEach(function (p) {
        shown += 1;
        out.push(h("div", { key: g.name + p.prefix, className: "sms-cat-item" + (p.status === "removed" || p.status === "unknown" ? " off" : "") },
          h("div", { className: "sms-col", style: { gap: "2px", flex: 1, minWidth: 0 } },
            h("div", { className: "sms-row" },
              h("code", null, p.prefix), h("span", null, p.name),
              p.status === "removed" ? h("span", { className: "sms-error" }, t("removedInBuild")) : null,
              p.status === "unknown" ? h("span", { className: "sms-warn" }, t("notInBuild")) : null,
              have[p.prefix] ? h(Tag, null, t("alreadyAdded")) : null),
            h("div", { className: "sms-muted sms-desc" }, p.description || "")),
          h(Btn, { variant: "outline", onClick: function () { props.onAdd(p); } }, t("add"))));
      });
    });
    return h("div", { className: "sms-col" },
      h("input", { className: "sms-input", placeholder: t("searchParamPlaceholder") + " " + (props.catalog ? "(" + props.catalog.count + ")" : ""), value: q, onChange: function (e) { setQ(e.target.value); } }),
      props.catalog && props.catalog.version ? h("div", { className: "sms-muted" }, t("catalogChecked", props.catalog.version)) : null,
      out.length ? h("div", { className: "sms-cat" }, out) : (needle ? h("div", { className: "sms-muted" }, t("noResults")) : null));
  }

  function ModelEditor(props) {
    const engines = (props.engines || []).filter(function (e) { return e.status !== "wip"; });
    const [d, setD] = useState(null);
    const [rows, setRows] = useState([]);
    const [baseSig, setBaseSig] = useState("");
    const [catalog, setCatalog] = useState(null);
    const [preview, setPreview] = useState(null);
    const [pending, setPending] = useState(false);
    const [msg, setMsg] = useState(null);
    const [manual, setManual] = useState({ key: "", value: "" });
    const [importText, setImportText] = useState("");
    const [showImport, setShowImport] = useState(false);

    function loadModel(name) {
      setMsg(null); setPreview(null);
      if (!name) {
        const dd = Object.assign({}, EMPTY);
        setD(dd); setRows([]); setBaseSig(sig(dd, []));
        return Promise.resolve();
      }
      return get("/models/detail?name=" + encodeURIComponent(name)).then(function (r) {
        if (!r || r.ok === false) { setMsg({ kind: "bad", text: (r && r.error) || t("couldNotLoad") }); return; }
        const dd = draftFrom(r.model); const rr = withIds(r.rows);
        setD(dd); setRows(rr); setBaseSig(sig(dd, rr));
        setPreview(r);
      }).catch(function (e) { setMsg({ kind: "bad", text: errText(e) }); });
    }
    useEffect(function () { loadModel(props.name); }, [props.name]);
    useEffect(function () {
      const engine = d ? d.engine : "llamacpp";
      get("/params/catalog?engine=" + encodeURIComponent(engine) + (props.name ? "&model=" + encodeURIComponent(props.name) : ""))
        .then(function (c) { setCatalog(c && c.ok ? c : null); }).catch(function () { setCatalog(null); });
    }, [props.name, d ? d.engine : ""]);

    if (!d) return h("div", { className: "sms-muted" }, msg ? msg.text : t("loading"));

    const dirty = sig(d, rows) !== baseSig;
    const byPrefix = {};
    (catalog ? catalog.groups || [] : []).forEach(function (g) { (g.parameters || []).forEach(function (p) { byPrefix[p.prefix] = p; }); });
    const have = {};
    rows.forEach(function (r) { have[r.key] = true; });

    function set(k) { return function (e) { const v = e.target.value; setD(function (o) { const n = Object.assign({}, o); n[k] = v; return n; }); }; }
    function setRow(id, patch) { setRows(function (rs) { return rs.map(function (r) { return r.id === id ? Object.assign({}, r, patch) : r; }); }); }
    function removeRow(id) { setRows(function (rs) { return rs.filter(function (r) { return r.id !== id; }); }); }
    function move(id, delta) {
      setRows(function (rs) {
        const i = rs.findIndex(function (r) { return r.id === id; }); const j = i + delta;
        if (i < 0 || j < 0 || j >= rs.length) return rs;
        const n = rs.slice(); const tmp = n[i]; n[i] = n[j]; n[j] = tmp; return n;
      });
    }
    function addParam(key, value) {
      key = String(key || "").trim();
      if (!/^-/.test(key)) { setMsg({ kind: "bad", text: t("paramMustStartWithDash") }); return false; }
      if (have[key] && !window.confirm(t("confirmAddAgain", key))) return false;
      setRows(function (rs) { return rs.concat([mkRow(key, value)]); }); setMsg(null);
      return true;
    }
    function addFromCatalog(p) {
      const val = p.type === "checkbox" ? null : (p.type === "select" ? String(p.default || (p.options || [])[0] || "") : String(p.default == null ? "" : p.default));
      addParam(p.prefix, val);
    }
    function run(path, body, after) {
      setPending(true); setMsg(null);
      return post(path, body).then(function (r) {
        if (r && r.ok === false) { setMsg({ kind: "bad", text: r.error || t("backendRejected") }); return r; }
        return after(r);
      }).catch(function (e) { setMsg({ kind: "bad", text: errText(e) }); })
        .then(function (r) { setPending(false); return r; });
    }
    // Shared context: what saving this draft will do to the other models' -c.
    function ctxEffectText() {
      if (!d) return null;
      const engine = d.engine || "llamacpp";
      const others = (props.models || []).filter(function (m) {
        return m.name !== props.name && m.name !== d.name && (m.engine || "llamacpp") === engine;
      });
      const ctxRow = rows.filter(function (r) { return r.key === "-c" || r.key === "--ctx-size"; })[0];
      const draftCtx = ctxRow ? String(ctxRow.value == null ? "" : ctxRow.value).trim() : "";
      if (draftCtx) {
        const hit = others.filter(function (m) { return String(m.ctx || "") !== draftCtx; }).map(function (m) { return m.name; });
        return hit.length ? t("ctxWillApply", draftCtx, hit.join(", ")) : null;
      }
      const shared = (others.filter(function (m) { return m.name === props.primary && m.ctx; })[0]
        || others.filter(function (m) { return m.ctx; })[0] || {}).ctx;
      return shared ? t("ctxWillInherit", shared) : null;
    }

    function doPreview() {
      return run("/models/preview", bodyOf(d, rows), function (r) { setPreview(r); return r; });
    }
    function doSave() {
      return run("/models", bodyOf(d, rows), function (r) {
        const w = (r && r.warnings) || [];
        const notes = (r && r.notes) || [];
        setMsg(w.length ? { kind: "warn", text: t("savedWithWarnings", w.concat(notes).join(" · ")) }
          : { kind: "ok", text: t("modelSaved", r.name) + (notes.length ? " " + notes.join(" · ") + "." : "") });
        return Promise.resolve(props.reload()).then(function () {
          if (props.name !== r.name) props.onSelect(r.name); else loadModel(r.name);
          return r;
        });
      });
    }
    function doImport() {
      return run("/models/parse_cmd", { cmdline: importText }, function (r) {
        if (rows.length && !window.confirm(t("confirmImportReplace"))) return r;
        setD(function (o) {
          return Object.assign({}, o, { gguf: r.gguf || o.gguf, mmproj: r.mmproj || "", bin: r.bin || o.bin,
            env: Object.keys(r.env || {}).length ? envToText(r.env) : o.env });
        });
        setRows(withIds(r.rows)); setShowImport(false); setImportText("");
        setMsg({ kind: "ok", text: t("importedLine", r.rows.length) + ((r.notes || []).length ? " " + r.notes.join(" · ") : "") });
        return r;
      });
    }
    function doDuplicate() {
      const n = window.prompt(t("duplicatePrompt", props.name), props.name + t("copySuffix"));
      if (!n) return;
      run("/models/duplicate", { name: props.name, new_name: n.trim() }, function (r) {
        return Promise.resolve(props.reload()).then(function () { props.onSelect(r.name); return r; });
      });
    }
    function doDelete() {
      if (!window.confirm(t("confirmDeleteModel", props.name))) return;
      run("/models/delete", { name: props.name }, function (r) {
        return Promise.resolve(props.reload()).then(function () { props.onClose(); return r; });
      });
    }

    const pv = preview;
    const ctxEffect = ctxEffectText();
    return h("div", { className: "sms-col" },
      h("div", { className: "sms-editor-head" },
        h("label", { className: "sms-field" }, h("span", { className: "sms-cap" }, t("nameLabel") + (dirty ? " ●" : "")),
          h("input", { className: "sms-input", value: d.name, onChange: set("name"), maxLength: 64, placeholder: t("namePlaceholder"), title: t("nameRules") })),
        h("label", { className: "sms-field" }, h("span", { className: "sms-cap" }, t("engineLabel")),
          h("select", { className: "sms-select", value: d.engine, onChange: set("engine") },
            (props.engines || []).map(function (e) { return h("option", { key: e.id, value: e.id, disabled: e.status === "wip" }, e.label + (e.status === "wip" ? " (" + t("underConstructionShort") + ")" : "")); }))),
        h("label", { className: "sms-field" }, h("span", { className: "sms-cap" }, t("roleLabel")),
          h("input", { className: "sms-input", value: d.role, onChange: set("role"), placeholder: t("rolePlaceholder") })),
        h("label", { className: "sms-field" }, h("span", { className: "sms-cap" }, t("minVramLabel")),
          h("input", { className: "sms-input", value: d.min_free_vram_gb, onChange: set("min_free_vram_gb"), inputMode: "decimal" })),
        h("label", { className: "sms-field" }, h("span", { className: "sms-cap" }, t("loadTimeoutLabel")),
          h("input", { className: "sms-input", value: d.load_timeout_s, onChange: set("load_timeout_s"), inputMode: "numeric" }))),
      h("label", { className: "sms-field" }, h("span", { className: "sms-cap" }, t("ggufLabel")),
        h("input", { className: "sms-input mono", value: d.gguf, onChange: set("gguf"), placeholder: t("ggufPlaceholder") })),
      h("label", { className: "sms-field" }, h("span", { className: "sms-cap" }, t("mmprojLabel")),
        h("input", { className: "sms-input mono", value: d.mmproj, onChange: set("mmproj"), placeholder: t("mmprojPlaceholder") })),
      h("div", { className: "sms-form" },
        h("label", { className: "sms-field" }, h("span", { className: "sms-cap" }, t("modelBinaryLabel")),
          h("input", { className: "sms-input mono", value: d.bin, onChange: set("bin"), placeholder: t("modelBinaryPlaceholder") })),
        h("label", { className: "sms-field" }, h("span", { className: "sms-cap" }, t("envVarsLabel")),
          h("textarea", { className: "sms-input mono", rows: 2, value: d.env, onChange: set("env"), placeholder: "MTMD_BACKEND_DEVICE=cuda1" }))),
      h("label", { className: "sms-field" }, h("span", { className: "sms-cap" }, t("descriptionLabel")),
        h("input", { className: "sms-input", value: d.description, onChange: set("description") })),

      h("div", { className: "sms-row" },
        h("strong", { className: "grow" }, t("launchParams", rows.length)),
        h("button", { className: "sms-link", onClick: function () { setShowImport(!showImport); } }, showImport ? t("closeImport") : t("importCmdline"))),
      showImport ? h("div", { className: "sms-col" },
        h("textarea", { className: "sms-input mono", rows: 3, value: importText, onChange: function (e) { setImportText(e.target.value); }, placeholder: t("importPlaceholder") }),
        h("div", { className: "sms-row" }, h(Btn, { disabled: pending || !importText.trim(), onClick: doImport }, t("importBtn")))) : null,
      h("div", { className: "sms-rows" },
        rows.length === 0 ? h("div", { className: "sms-muted" }, t("noOwnParams")) : null,
        h("div", { className: "sms-muted" }, t("ctxSharedHint")),
        ctxEffect ? h("div", { className: "sms-warn" }, ctxEffect) : null,
        rows.map(function (r, i) {
          const info = byPrefix[r.key];
          const bad = pv && (pv.unsupported || []).indexOf(r.key) >= 0;
          return h("div", { key: r.id, className: "sms-prow" + (bad ? " bad" : "") },
            h("input", { className: "sms-input mono sms-pkey", value: r.key, title: info ? info.name + " — " + (info.description || "") : "", onChange: function (e) { setRow(r.id, { key: e.target.value }); } }),
            h(ParamValue, { row: r, info: info, onChange: function (v) { setRow(r.id, { value: v }); }, onRemove: function () { removeRow(r.id); } }),
            (r.key === "-c" || r.key === "--ctx-size") ? h("span", { className: "sms-muted", title: t("ctxSharedHint") }, t("ctxSharedBadge")) : null,
            h("button", { className: "sms-link", title: t("moveUp"), disabled: i === 0, onClick: function () { move(r.id, -1); } }, "↑"),
            h("button", { className: "sms-link", title: t("moveDown"), disabled: i === rows.length - 1, onClick: function () { move(r.id, 1); } }, "↓"),
            h("button", { className: "sms-link sms-x", title: t("removeParam", r.key), onClick: function () { removeRow(r.id); } }, "✕"));
        })),
      h("div", { className: "sms-row" },
        h("input", { className: "sms-input mono", style: { width: "170px" }, placeholder: t("paramPlaceholder"), value: manual.key, onChange: function (e) { setManual({ key: e.target.value, value: manual.value }); } }),
        h("input", { className: "sms-input mono", style: { flex: 1 }, placeholder: t("valuePlaceholder"), value: manual.value, onChange: function (e) { setManual({ key: manual.key, value: e.target.value }); } }),
        h(Btn, { variant: "outline", disabled: !manual.key.trim(), onClick: function () {
          if (addParam(manual.key, manual.value.trim() === "" ? null : manual.value.trim())) setManual({ key: "", value: "" });
        } }, t("add"))),
      h(Catalog, { catalog: catalog, have: have, onAdd: addFromCatalog }),

      h("div", { className: "sms-row" },
        h(Btn, { disabled: pending || !dirty, onClick: doSave }, t("save")),
        h(Btn, { variant: "outline", disabled: pending, onClick: doPreview }, t("previewValidate")),
        h(Btn, { variant: "outline", disabled: pending || !dirty, onClick: function () { loadModel(props.name); } }, t("reset")),
        props.name ? h(Btn, { variant: "outline", disabled: pending, onClick: doDuplicate }, t("duplicate")) : null,
        props.name && props.name !== props.primary ? h("button", { className: "sms-link", disabled: pending, onClick: doDelete }, t("deleteLink")) : null,
        h("span", { className: "grow" }),
        h("button", { className: "sms-link", onClick: function () {
          if (!dirty || window.confirm(t("confirmCloseUnsaved"))) props.onClose();
        } }, t("close"))),
      h(Msg, { msg: msg }),
      pv && pv.ok !== false ? h("div", { className: "sms-col" },
        h("div", { className: "sms-row" },
          h("span", { className: "sms-dot " + (pv.validate && pv.validate.ok ? "good" : "bad") }),
          h("span", null, pv.validate && pv.validate.ok ? t("binaryAccepts", pv.build) : t("rejected", (pv.validate && pv.validate.error) || "?")),
          dirty ? h("span", { className: "sms-muted" }, t("previewOfSaved")) : null),
        (pv.unsupported || []).length ? h("div", { className: "sms-warn" }, t("unrecognized", pv.unsupported.join(", "))) : null,
        (pv.fix_notes || []).length ? h("div", { className: "sms-col" },
          h("div", { className: "sms-warn" }, t("suggestedFixes", pv.fix_notes.join(" · "))),
          pv.fixed_rows ? h("div", null, h(Btn, { variant: "outline", onClick: function () { setRows(withIds(pv.fixed_rows)); setMsg({ kind: "ok", text: t("fixesApplied") }); } }, t("applyFixes"))) : null) : null,
        h("pre", { className: "sms-log sms-cmd" }, pv.cmdline || "")) : null);
  }

  // Sampling-param key (backend value) -> i18n key of its label.
  const PARAM_LABELS = {
    temperature: "paramTemperature", top_p: "paramTopP", top_k: "paramTopK", min_p: "paramMinP",
    presence_penalty: "paramPresencePenalty", repeat_penalty: "paramRepeatPenalty"
  };
  function paramError(spec, raw) {
    const text = String(raw == null ? "" : raw).trim().replace(",", ".");
    const n = Number(text);
    if (text === "" || !isFinite(n)) return t("notANumber");
    if (spec.integer && Math.floor(n) !== n) return t("mustBeInteger");
    if (n < spec.min || n > spec.max) return t("outOfRange", spec.min, spec.max);
    return null;
  }

  // Sampling params of one model + its presets. Values are applied when the
  // model is next loaded: base args, overridden by the active preset.
  function Params(props) {
    const models = props.models, specs = props.specs;
    const [picked, setPicked] = useState("");
    const [edit, setEdit] = useState(null); // {key, vals}: unsaved edits for one model+preset
    const [name, setName] = useState("");
    const [pending, setPending] = useState(false);
    const [msg, setMsg] = useState(null);

    const model = picked && models.some(function (m) { return m.name === picked; })
      ? picked : (props.current || props.primary || (models[0] && models[0].name) || "");
    const entry = models.filter(function (m) { return m.name === model; })[0] || null;
    if (!entry) return h("div", { className: "sms-muted" }, t("noModels"));

    const preset = entry.active_preset || "";
    const key = model + "|" + preset;
    const saved = entry.params || {};
    const vals = edit && edit.key === key ? edit.vals : saved;
    const dirty = specs.some(function (sp) { return String(vals[sp.key]) !== String(saved[sp.key]); });
    const invalid = specs.some(function (sp) { return paramError(sp, vals[sp.key]); });
    const presetNames = Object.keys(entry.presets || {});
    const newName = name.trim();

    function send(path, body, okText) {
      setPending(true); setMsg(null);
      return post(path, body).then(function (r) {
        if (r && r.ok === false) setMsg({ kind: "bad", text: r.error || t("backendRejected") });
        else { setEdit(null); setMsg({ kind: "ok", text: okText }); }
        return r;
      }).catch(function (e) { setMsg({ kind: "bad", text: String((e && e.message) || e) }); return null; })
        .then(function (r) { return Promise.resolve(props.reload()).then(function () { setPending(false); return r; }); });
    }
    function setVal(k) {
      return function (e) {
        const next = Object.assign({}, vals); next[k] = e.target.value;
        setEdit({ key: key, vals: next }); setMsg(null);
      };
    }

    return h("div", { className: "sms-col" },
      h("div", { className: "sms-params-head" },
        h("label", { className: "sms-field" },
          h("span", { className: "sms-cap" }, t("modelLabel")),
          h("select", {
            className: "sms-select", value: model, disabled: pending,
            onChange: function (e) { setPicked(e.target.value); setEdit(null); setMsg(null); }
          }, models.map(function (m) {
            return h("option", { key: m.name, value: m.name }, m.name + (m.name === props.current ? " (" + t("loadedSuffix") + ")" : ""));
          }))),
        h("label", { className: "sms-field" },
          h("span", { className: "sms-cap" }, t("activePresetLabel")),
          h("select", {
            className: "sms-select", value: preset, disabled: pending,
            onChange: function (e) {
              const p = e.target.value;
              send("/params/preset/select", { model: model, preset: p },
                p ? t("presetActiveMsg", p, model)
                  : t("noPresetMsg", model));
            }
          },
            h("option", { value: "" }, t("basePreset")),
            presetNames.map(function (p) { return h("option", { key: p, value: p }, p); })))),

      h("div", { className: "sms-params" }, specs.map(function (sp) {
        const err = paramError(sp, vals[sp.key]);
        return h("label", { key: sp.key, className: "sms-field", title: sp.flag + " · " + sp.min + "–" + sp.max },
          h("span", { className: "sms-cap" }, PARAM_LABELS[sp.key] ? t(PARAM_LABELS[sp.key]) : sp.key),
          h("input", {
            className: "sms-input" + (err ? " bad" : ""), type: "number", inputMode: "decimal",
            min: sp.min, max: sp.max, step: sp.integer ? 1 : 0.05,
            value: vals[sp.key] == null ? "" : vals[sp.key], disabled: pending,
            "aria-invalid": err ? "true" : "false", onChange: setVal(sp.key)
          }),
          err ? h("span", { className: "sms-error" }, err) : null);
      })),

      h("div", { className: "sms-row" },
        h(Btn, {
          disabled: pending || !dirty || invalid,
          onClick: function () {
            send("/params/save", { model: model, params: vals, preset: preset },
              preset ? t("presetUpdated", preset) : t("baseParamsSaved", model));
          }
        }, preset ? t("saveToPreset", preset) : t("saveBaseParams")),
        dirty ? h(Btn, { variant: "outline", disabled: pending, onClick: function () { setEdit(null); setMsg(null); } }, t("discard")) : null,
        preset ? h("button", {
          className: "sms-link", disabled: pending,
          onClick: function () {
            if (window.confirm(t("confirmDeletePreset", preset, model)))
              send("/params/preset/delete", { model: model, preset: preset }, t("presetDeleted", preset));
          }
        }, t("deletePreset")) : null),

      h("div", { className: "sms-row" },
        h("input", {
          className: "sms-input", style: { flex: 1 }, placeholder: t("newPresetPlaceholder"), maxLength: 40,
          value: name, disabled: pending, onChange: function (e) { setName(e.target.value); }
        }),
        h(Btn, {
          variant: "outline", disabled: pending || !newName || invalid,
          onClick: function () {
            const exists = presetNames.indexOf(newName) >= 0;
            if (exists && !window.confirm(t("confirmOverwritePreset", newName))) return;
            send("/params/save", { model: model, params: vals, preset: newName },
              t("presetSavedActive", newName)).then(function (r) { if (r && r.ok) setName(""); });
          }
        }, t("saveAsPreset"))),

      h("div", { className: "sms-muted" }, t("paramsApplyNote", model === props.current)),
      msg ? h("div", { className: msg.kind === "bad" ? "sms-error" : "sms-ok" }, msg.text) : null);
  }

  // What gpu_job.sh does with the LLM while a VRAM-heavy job runs (setting
  // gpu_job_mode). One option per mode; the backend says what the selected
  // one means on this machine right now.
  function GpuJobModes(props) {
    const g = props.data || {};
    const modes = g.modes || [];
    return h("div", { className: "sms-col" },
      h("div", { className: "sms-muted" }, t("gpuModeNote")),
      h("div", { className: "sms-modes", role: "radiogroup", "aria-label": t("gpuModeAria") },
        modes.map(function (m) {
          const on = m.id === g.mode;
          return h("button", {
            key: m.id, type: "button", role: "radio", "aria-checked": on ? "true" : "false",
            className: "sms-mode" + (on ? " on" : ""), disabled: !!props.disabled,
            onClick: function () { if (!on) props.onPick(m); }
          },
            h("span", { className: "sms-mode-name" }, m.label),
            h("span", { className: "sms-mode-desc" }, m.description));
        })),
      g.summary ? h("div", { className: g.layout && g.layout.degraded ? "sms-warn" : "sms-muted" },
        t("gpuModeNow") + " " + g.summary) : null);
  }

  // Behaviour settings (the same set the Desktop pane shows): one row each,
  // saved on change through POST /settings.
  function SetRow(props) {
    return h("div", { className: "sms-set" },
      h("div", { className: "sms-set-text" },
        h("span", { className: "sms-set-label", id: props.id }, props.label),
        props.hint ? h("span", { className: "sms-muted" }, props.hint) : null),
      h("div", { className: "sms-set-control" }, props.children));
  }
  function Toggle(props) {
    return h("button", {
      type: "button", role: "switch", className: "sms-switch sms-switch-sm",
      "aria-checked": props.value ? "true" : "false", "aria-labelledby": props.labelId,
      title: props.value ? t("on") : t("off"), disabled: !!props.disabled,
      onClick: function () { props.onChange(!props.value); }
    }, h("span", null));
  }
  function Settings(props) {
    const s = props.settings || {};
    const [idle, setIdle] = useState(null); // text being edited, null = show the saved value
    const idleText = idle == null ? String(s.idle_restore_min == null ? "" : s.idle_restore_min) : idle;
    const idleNum = /^\d+$/.test(idleText.trim()) ? parseInt(idleText, 10) : NaN;
    const idleBad = idle != null && !(idleNum >= 0 && idleNum <= 720);
    function save(key, value) { return props.onSave(key, value); }
    function commitIdle() {
      if (idle == null) return;
      if (idleBad) return;
      if (idleNum === s.idle_restore_min) { setIdle(null); return; }
      Promise.resolve(save("idle_restore_min", idleNum)).then(function () { setIdle(null); });
    }
    return h("div", { className: "sms-col" },
      h(SetRow, { id: "sms-set-after", label: t("setAfterJob"), hint: t("setAfterJobHint") },
        h("div", { className: "sms-seg", role: "radiogroup", "aria-labelledby": "sms-set-after" },
          [["small", t("setAfterSmall")], ["primary", t("setAfterPrimary")]].map(function (o) {
            const on = s.after_gpu_job === o[0];
            return h("button", {
              key: o[0], type: "button", role: "radio", "aria-checked": on ? "true" : "false",
              className: on ? "on" : "", disabled: !!props.disabled,
              onClick: function () { if (!on) save("after_gpu_job", o[0]); }
            }, o[1]);
          }))),
      h(SetRow, { id: "sms-set-idle", label: t("setIdle"), hint: idleBad ? null : t("setIdleHint") },
        h("input", {
          className: "sms-input sms-num" + (idleBad ? " bad" : ""), inputMode: "numeric", value: idleText,
          "aria-labelledby": "sms-set-idle", disabled: !!props.disabled,
          onChange: function (e) { setIdle(e.target.value); },
          onBlur: commitIdle,
          onKeyDown: function (e) { if (e.key === "Enter") commitIdle(); if (e.key === "Escape") setIdle(null); }
        }),
        h("span", { className: "sms-muted" }, t("setIdleUnit"))),
      idleBad ? h("div", { className: "sms-error" }, t("setIdleBad")) : null,
      h(SetRow, { id: "sms-set-recover", label: t("setRecover"), hint: t("setRecoverHint") },
        h(Toggle, { labelId: "sms-set-recover", value: !!s.auto_recover, disabled: props.disabled,
          onChange: function (v) { save("auto_recover", v); } })),
      h(SetRow, { id: "sms-set-autostart", label: t("setAutostart"), hint: t("setAutostartHint") },
        h(Toggle, { labelId: "sms-set-autostart", value: !!s.autostart_primary, disabled: props.disabled,
          onChange: function (v) { save("autostart_primary", v); } })));
  }

  // SmartModelProvider: the model-provider half of this plugin. It is only
  // written to disk when the user asks for it here (or with the CLI).
  function Provider(props) {
    const pv = props.provider || {};
    const [pending, setPending] = useState(false);
    const [msg, setMsg] = useState(null);
    function send(path, okText) {
      setPending(true); setMsg(null);
      return post(path, {}).then(function (r) {
        if (r && r.ok === false) setMsg({ kind: "bad", text: r.error || t("backendRejected") });
        else setMsg({ kind: "ok", text: okText });
      }).catch(function (e) { setMsg({ kind: "bad", text: String((e && e.message) || e) }); })
        .then(function () { return Promise.resolve(props.reload()); })
        .then(function () { setPending(false); });
    }
    const models = pv.models || [];
    return h("div", { className: "sms-col" },
      h("div", { className: "sms-row" },
        h("span", { className: "sms-dot " + (pv.installed ? (pv.in_sync ? "good" : "warn") : "") }),
        h("span", null, pv.installed ? (pv.in_sync ? t("installed") : t("installedOutdated")) : pv.foreign ? t("folderTaken") : t("notInstalled")),
        pv.installed ? h(Tag, null, pv.id) : null),
      pv.error ? h("div", { className: "sms-error" }, pv.error) : null,
      pv.foreign
        ? h("div", { className: "sms-warn" }, t("foreignFolder", pv.path))
        : null,
      !pv.installed && !pv.foreign ? h("div", { className: "sms-col" },
        h("div", { className: "sms-muted" }, t("providerExplain")),
        h("div", { className: "sms-row" },
          h(Btn, { disabled: pending, onClick: function () { send("/provider/install", t("providerInstalled")); } }, pending ? "…" : t("installProvider"))),
        h("div", { className: "sms-muted" }, t("createsFolder", pv.path || "plugins/model-providers/smart-model-provider"))) : null,
      pv.installed ? h("div", { className: "sms-col" },
        models.map(function (m) {
          return h("div", { key: m.name, className: "sms-item" },
            h("span", { className: "grow" }, m.name),
            m.ctx ? h("span", { className: "sms-muted" }, t("tokens", m.ctx.toLocaleString(LANG))) : null);
        }),
        h("div", { className: "sms-muted" }, t("providerHowTo")),
        h("div", { className: "sms-row" },
          !pv.in_sync ? h(Btn, { disabled: pending, onClick: function () { send("/provider/install", t("providerUpdated")); } }, t("update")) : null,
          h("button", {
            className: "sms-link", disabled: pending,
            onClick: function () {
              if (window.confirm(t("confirmRemoveProvider"))) send("/provider/remove", t("providerRemoved"));
            }
          }, t("removeLink")))) : null,
      msg ? h("div", { className: msg.kind === "bad" ? "sms-error" : "sms-ok" }, msg.text) : null);
  }

  function Page() {
    const i18n = SDK.useI18n ? SDK.useI18n() : null;
    LANG = langOf((i18n && i18n.locale) || navigator.language);
    const [data, setData] = useState(null);
    const [loadErr, setLoadErr] = useState(null);
    const [msg, setMsg] = useState(null); // {kind: 'ok'|'bad', text}
    const [pending, setPending] = useState(null);
    const [editor, setEditor] = useState(null); // {name: string|null} - null name = new model
    const [, setTick] = useState(0);
    const alive = useRef(true);
    const editorRef = useRef(null);

    const load = useCallback(function () {
      return get("/state").then(function (d) {
        if (!alive.current) return;
        setData(d); setLoadErr(null);
      }).catch(function (e) { if (alive.current) setLoadErr(errText(e)); });
    }, []);

    const status = (data && data.status) || {};
    const swap = status.swap || {};
    const state = swap.state || "idle";
    const gpuJob = status.gpu_job || null;
    const busy = BUSY.indexOf(state) >= 0;

    useEffect(function () {
      alive.current = true;
      load();
      const poll = setInterval(load, busy ? 1000 : 5000);
      const c = setInterval(function () { setTick(function (n) { return n + 1; }); }, 1000);
      return function () { alive.current = false; clearInterval(poll); clearInterval(c); };
    }, [load, busy]);

    function act(key, path, body, okText) {
      setPending(key); setMsg(null);
      return post(path, body).then(function (r) {
        if (r && r.ok === false) setMsg({ kind: "bad", text: r.error || t("backendRejected") });
        else if (okText) setMsg({ kind: "ok", text: typeof okText === "function" ? okText(r) : okText });
        return r;
      }).catch(function (e) {
        setMsg({ kind: "bad", text: errText(e) });
        return null;
      }).then(function (r) { setPending(null); load(); return r; });
    }
    function openEditor(name) {
      setEditor({ name: name });
      setTimeout(function () { if (editorRef.current && editorRef.current.scrollIntoView) editorRef.current.scrollIntoView({ behavior: "smooth", block: "start" }); }, 50);
    }

    if (!data) {
      return h("div", { className: "sms-page" },
        h("p", { className: loadErr ? "sms-error" : "sms-muted" }, loadErr ? t("couldNotLoadState", loadErr) : t("loading")));
    }

    const ui = data.ui || {};
    const slots = ui.slots || {};
    const models = data.models || [];
    const health = data.health || {};
    const engines = data.engines || [];
    const activeEngine = engines.filter(function (e) { return e.active; })[0] || {};
    const managed = activeEngine.managed !== false;
    const llm = status.llm || {};
    const server = status.server || {};
    const comfy = status.comfyui || {};
    const vram = Array.isArray(status.vram) ? status.vram : [];
    const last = swap.last_result || null;
    const graceLeft = state === "grace" && swap.grace_until
      ? Math.max(0, Math.ceil(swap.grace_until - Date.now() / 1000)) : null;
    const failText = state === "failed"
      ? (swap.stale ? t("runnerDied") : (last && last.detail && last.detail.error) || t("lastSwapFailed"))
      : null;
    const stateText = graceLeft != null ? t("cutoverIn", graceLeft, swap.target || "…")
      : busy ? state + " → " + (swap.target || "…") : state;
    const noSwap = busy || !managed || !!pending;
    const primary = (data.config && data.config.primary_model) || slots.primary || "";
    const secondary = slots.secondary || "";
    // Switch position: right = the secondary is (or is about to be) the loaded model.
    const onSecondary = busy
      ? swap.action === "swap"
      : !!llm.current && llm.current !== primary;

    function toggle() {
      if (onSecondary || !llm.current) act("restore", "/restore", {}, t("switchingToPrimary", primary));
      else act("swap", "/swap", { model: secondary }, t("switchingToSecondary", secondary));
    }
    function pick(slot) {
      return function (e) {
        act("slot", "/slots", { slot: slot, model: e.target.value }, function (r) {
          return r && r.exchanged ? t("slotsExchanged") : (slot === "primary" ? t("primarySet", e.target.value) : t("secondarySet", e.target.value));
        });
      };
    }
    function slotSelect(slot, value) {
      return h("select", { className: "sms-select", value: value, disabled: busy || !!pending, onChange: pick(slot) },
        h("option", { value: "", disabled: true }, t("choose")),
        models.map(function (m) { return h("option", { key: m.name, value: m.name }, m.name); }));
    }

    function dryRun() {
      setPending("dry"); setMsg(null);
      post("/swap", { model: slots.secondary, dry_run: true }).then(function (r) {
        const pf = (r && r.preflight) || {};
        const bad = (pf.checks || []).filter(function (c) { return !c.ok; })[0];
        if (r && r.ok) setMsg({ kind: "ok", text: t("dryRunOk", slots.secondary, pf.available_gb) });
        else setMsg({ kind: "bad", text: t("dryRunFail", bad ? bad.check + " — " + bad.detail : (r && r.error) || (r && r.busy ? t("swapInProgress") : t("failed"))) });
      }).catch(function (e) { setMsg({ kind: "bad", text: errText(e) }); })
        .then(function () { setPending(null); });
    }

    return h("div", { className: "sms-page" },
      loadErr ? h("p", { className: "sms-error" }, t("noBackend", loadErr)) : null,

      h("div", { className: "sms-grid" },
        h(Card, { title: t("activeModelTitle") },
          h("div", { className: "sms-row" },
            h("span", { className: "sms-dot " + (llm.up ? tone(state) : gpuJob ? "warn" : "bad") }),
            h("span", { className: "sms-model" }, llm.current || (llm.up ? t("unregisteredModel") : state === "starting" ? t("startingUp") : gpuJob ? t("gpuJobUnloaded") : t("serverDown")))),
          gpuJob ? h("div", { className: "sms-warn" }, t(llm.up ? "gpuJobRunningNote" : "gpuJobUnloadedNote", gpuJob.job, Math.floor((gpuJob.elapsed_s || 0) / 60))) : null,
          h("div", { className: "sms-row sms-muted" },
            h("span", null, t("stateText", stateText)),
            activeEngine.label ? h("span", null, t("engineText", activeEngine.label)) : null,
            server.summary ? h("span", null, t("serverText", server.summary)) : null),
          failText ? h("div", { className: "sms-error" }, failText) : null,
          h("div", { className: "sms-switch-row" },
            h("div", { className: "sms-side" + (!onSecondary && llm.current === primary ? " on" : "") },
              h("span", { className: "sms-cap" }, t("primary")),
              slotSelect("primary", primary)),
            h("button", {
              type: "button", role: "switch", className: "sms-switch",
              "aria-checked": onSecondary ? "true" : "false",
              "aria-label": t("toggleAria"),
              title: onSecondary ? t("backToPrimary") : (llm.current ? t("switchToSecondary") : t("loadPrimary")),
              disabled: noSwap || (!onSecondary && llm.current && (!secondary || secondary === primary)),
              onClick: toggle
            }, h("span", null)),
            h("div", { className: "sms-side right" + (onSecondary && !busy ? " on" : "") },
              h("span", { className: "sms-cap" }, t("secondary")),
              slotSelect("secondary", secondary))),
          h("div", { className: "sms-row" },
            h(Btn, { variant: "outline", disabled: noSwap || !secondary, onClick: dryRun }, pending === "dry" ? "…" : t("testSwap")),
            h("span", { className: "sms-muted" }, t("selectorNote"))),
          !managed ? h("div", { className: "sms-muted" }, t("engineWipNote")) : null,
          h(Msg, { msg: msg })),

        h(Card, { title: t("vramTitle") },
          vram.length === 0 ? h("div", { className: "sms-muted" }, t("noNvidiaSmi")) : null,
          vram.map(function (g) {
            const pct = g.total_mib ? Math.round((g.used_mib / g.total_mib) * 100) : 0;
            return h("div", { key: g.gpu, className: "sms-col", style: { gap: "4px" } },
              h("div", { className: "sms-row sms-muted" },
                h("span", { style: { flex: 1 } }, "GPU " + g.gpu + " · " + g.name),
                h("span", null, gb(g.used_mib) + " / " + gb(g.total_mib))),
              h("div", { className: "sms-bar" }, h("span", { style: { width: pct + "%" } })));
          }),
          h("div", { className: "sms-row sms-muted" },
            h("span", null, t("freeText", status.total_free_gb != null ? status.total_free_gb + " GB" : "?")),
            h("span", null, "ComfyUI: " + (comfy.up ? t("comfyUp") : t("comfyDown")))))),

      h(Card, { title: t("gpuModeTitle") },
        h(GpuJobModes, {
          data: data.gpu_job_modes, disabled: !!pending,
          onPick: function (m) { act("gpumode", "/gpu_job/mode", { mode: m.id }, t("gpuModeSaved", m.label)); }
        })),

      h(Card, { title: t("settingsTitle") },
        h(Settings, {
          settings: data.settings, disabled: !!pending,
          onSave: function (key, value) { return act("setting", "/settings", { key: key, value: value }, t("setSaved")); }
        })),

      h("div", { className: "sms-grid" },
        h(Card, { title: t("enginesTitle") },
          h(Engines, { engines: engines, health: health, reload: load })),

        h(Card, { title: t("modelsTitle", models.length) },
          models.map(function (m) {
            return h("div", { key: m.name, className: "sms-item" + (editor && editor.name === m.name ? " active" : "") },
              h("span", { className: "sms-dot " + (m.exists ? "good" : "warn"), title: m.exists ? t("ggufFound") : t("ggufNotFound") }),
              h("span", { className: "grow", title: m.gguf }, m.name + (m.role ? " · " + m.role : "")),
              m.ctx ? h("span", { className: "sms-muted" }, Math.round(m.ctx / 1024) + "k ctx") : null,
              m.min_free_vram_gb ? h("span", { className: "sms-muted" }, m.min_free_vram_gb + " GB") : null,
              m.active_preset ? h(Tag, null, t("presetTag", m.active_preset)) : null,
              m.name === primary ? h(Tag, null, t("primaryTag")) : m.name === slots.secondary ? h(Tag, null, t("secondaryTag")) : null,
              h("button", { className: "sms-link", onClick: function () { openEditor(m.name); } }, t("edit")));
          }),
          h("div", { className: "sms-row" },
            h(Btn, { variant: "outline", onClick: function () { openEditor(null); } }, t("newModel")),
            h("span", { className: "sms-muted" }, t("editorNote"))))),

      editor ? h("div", { ref: editorRef },
        h(Card, { title: editor.name ? t("editorTitle", editor.name) : t("newModel") },
          h(ModelEditor, {
            key: editor.name || "__new__", name: editor.name, engines: engines, primary: primary, reload: load, models: models,
            onSelect: function (n) { setEditor({ name: n }); }, onClose: function () { setEditor(null); }
          }))) : null,

      h("div", { className: "sms-grid" },
        h(Card, { title: t("samplingTitle") },
          h(Params, { models: models, specs: data.param_specs || [], current: llm.current || "", primary: primary, reload: load })),
        h(Card, { title: t("providerTitle") },
          h(Provider, { provider: data.provider, reload: load }))),

      h(Card, { title: t("runnerLogTitle") },
        h("pre", { className: "sms-log" }, (data.log_tail || []).join("\n") || t("noLogsYet"))));
  }

  window.__HERMES_PLUGINS__.register("smart-model-swap", Page);
})();
