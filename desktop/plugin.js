/**
 * Smart-Model-Swap — desktop UI for managing primary/secondary models and
 * SmartModelProvider: the inference engines it launches (llama.cpp managed;
 * Ollama, vLLM, LM Studio under construction) and a Launcher-style editor for
 * every model's launch parameters.
 *
 * Save as:  ~/.hermes/desktop-plugins/smart-model-swap/plugin.js
 * Reload with "Reload desktop plugins" from the desktop app (⌘K).
 * Backend:  the plugin ships dashboard/plugin_api.py (mounted at
 *           /api/plugins/smart-model-swap/ by the gateway — hot-mount on
 *           first request, no restart).
 */
import {
  Badge,
  Button,
  Codicon,
  GlyphSpinner,
  Separator,
  StatusDot,
  Tip,
  cn,
  haptic,
  host,
  useI18n,
  useQuery,
  useValue
} from '@hermes/plugin-sdk'
import { jsx, jsxs } from 'react/jsx-runtime'
import { useEffect, useState } from 'react'

// --------------------------------------------------------------------------- i18n (es / en)

const I18N = {
  es: {
    // helpers
    timeJustNow: 'hace un momento',
    timeMinutesAgo: s => `hace ${s} min`,
    timeHoursAgo: n => `hace ${n} h`,
    paramTemperature: 'Temperatura',
    errNotANumber: 'no es un número',
    errMustBeInteger: 'debe ser entero',
    errOutOfRange: (min, max) => `fuera de ${min}–${max}`,
    // common
    save: 'Guardar',
    close: 'Cerrar',
    closeLower: 'cerrar',
    failed: 'falló',
    // engines
    engineSaved: label => `${label} guardado`,
    underConstruction: '🚧 En construcción',
    activeBadge: 'activo',
    activeEngineFlash: label => `Motor activo: ${label}`,
    activate: 'activar',
    port: 'Puerto',
    configureEngine: 'Configurar motor',
    online: 'en línea',
    modelCount: n => `${n} modelo(s)`,
    noResponse: 'sin respuesta',
    binaryNotFound: 'Binario no encontrado: configura la ruta del motor.',
    engineBinaryLabel: 'Binario (llama-server o su carpeta)',
    envLabel: 'Entorno (KEY=valor por línea)',
    commonArgsLabel: 'Args comunes (todos los modelos)',
    test: 'Probar',
    // model editor
    couldNotLoad: 'no se pudo cargar',
    loading: 'Cargando…',
    paramMustStartWithDash: "El parámetro debe empezar por '-' o '--'.",
    confirmDuplicateParam: key => `'${key}' ya está en la lista. ¿Añadirlo otra vez?`,
    backendRejected: 'el backend rechazó la acción',
    savedWithWarnings: w => `Guardado. ${w}`,
    modelSaved: n => `Modelo '${n}' guardado`,
    confirmImportReplace: 'Se reemplazarán GGUF, mmproj, entorno y parámetros. ¿Continuar?',
    importedParams: (n, notes) => `Importados ${n} parámetros. ${notes}`,
    promptCopyName: n => `Nombre de la copia de '${n}':`,
    copyNameDefault: n => `${n}-copia`,
    confirmDeleteModel: n => `¿Eliminar '${n}' del registro?`,
    modelDeleted: n => `Modelo '${n}' eliminado`,
    editModelTitle: n => `Editar · ${n}`,
    newModel: 'Nuevo modelo',
    confirmCloseUnsaved: 'Hay cambios sin guardar. ¿Cerrar?',
    name: 'Nombre',
    namePlaceholder: 'p. ej. Qwen3.5-9B_Q5',
    nameRulesTip: 'Letras (mayúsculas/minúsculas), dígitos y . _ - + ( ) [ ] , — sin espacios, / ni :',
    engine: 'Motor',
    mmprojOptional: 'mmproj (opcional)',
    role: 'Rol',
    minVramGb: 'VRAM mín. GB',
    timeoutSeconds: 'Timeout s',
    modelBinaryLabel: 'Binario para este modelo (vacío = el del motor)',
    paramsCount: n => `Parámetros (${n})`,
    importLine: 'importar línea',
    importPlaceholder: 'llama-server -m /ruta/modelo.gguf -c 32768 --temp 0.7 …',
    import: 'Importar',
    noOwnParams: 'Sin parámetros propios: solo los args comunes del motor.',
    ctxSharedHint: 'El contexto (-c / --ctx-size) es compartido: al guardar se aplica a todos los modelos del motor (Hermes usa un único contexto).',
    ctxWillApply: (ctx, names) => `Al guardar, el contexto ${ctx} se aplicará también a: ${names}.`,
    ctxWillInherit: ctx => `Sin -c: al guardar se usará el contexto compartido (${ctx}).`,
    ctxSharedBadge: 'compartido',
    switchValue: 'switch',
    removeParam: key => `Quitar ${key}`,
    valuePlaceholder: 'valor (vacío = switch)',
    add: 'Añadir',
    searchCatalog: suffix => `Buscar en el catálogo${suffix}…`,
    noResults: 'Sin resultados.',
    validate: 'Validar',
    reset: 'Restablecer',
    duplicate: 'Duplicar',
    delete: 'Eliminar',
    binaryAcceptsLine: suffix => `El binario acepta la línea${suffix}`,
    rejected: err => `Rechazada: ${err}`,
    unrecognized: list => `No reconocidos: ${list}`,
    suggestions: s => `Sugerencias: ${s}`,
    apply: 'Aplicar',
    // model registry row
    primarySuffix: '  (primario)',
    secondarySuffix: '  (secundario)',
    editLaunchParams: 'Editar parámetros de arranque',
    // params + presets
    modelParams: 'Parámetros del modelo',
    unsaved: 'sin guardar',
    noModelsInRegistry: 'No hay modelos en el registro.',
    model: 'Modelo',
    loadedSuffix: n => `${n} (cargado)`,
    activePreset: 'Preset activo',
    presetActive: p => `Preset '${p}' activo.`,
    noPresetBase: 'Sin preset: parámetros base.',
    basePresetOption: 'Base (sin preset)',
    presetUpdated: p => `Preset '${p}' actualizado.`,
    baseParamsSaved: m => `Parámetros base de ${m} guardados.`,
    saveToPreset: p => `Guardar en '${p}'`,
    saveBase: 'Guardar base',
    discard: 'Descartar',
    confirmDeletePreset: (p, m) => `¿Eliminar el preset '${p}' de ${m}?`,
    presetDeleted: p => `Preset '${p}' eliminado.`,
    deletePreset: 'Eliminar preset',
    newPresetPlaceholder: 'nombre del nuevo preset',
    confirmOverwritePreset: p => `El preset '${p}' ya existe. ¿Sobrescribirlo?`,
    presetSavedActive: p => `Preset '${p}' guardado y activo.`,
    saveAsPreset: 'Guardar como preset',
    appliedNextLoad: 'Se aplican la próxima vez que se cargue el modelo',
    appliedNextLoadReload: '; para aplicarlos al que está cargado hay que recargarlo.',
    // SmartModelProvider
    providerInstalled: 'instalado',
    providerOutdated: 'desactualizado',
    providerFolderTaken: 'carpeta ocupada',
    providerNotInstalled: 'no instalado',
    providerForeign: p => `Ya existe ${p} y no lo creó este plugin; no se toca.`,
    providerExplain: 'Expone los modelos del registro como proveedor de Hermes, para elegirlos en /model con su contexto real.',
    providerInstalledMsg: 'SmartModelProvider instalado.',
    installProvider: 'Instalar SmartModelProvider',
    providerCreatesFolder: p => `Crea la carpeta ${p}. No funciona sin este plugin.`,
    providerHowTo:
      'Elígelo en /model → SmartModelProvider. Elegir ahí un modelo no lo carga; la lista llega a Hermes al reiniciar gateway o serve.',
    providerUpdatedMsg: 'SmartModelProvider actualizado.',
    update: 'Actualizar',
    confirmRemoveProvider: '¿Quitar SmartModelProvider?',
    providerRemovedMsg: 'SmartModelProvider quitado.',
    remove: 'Quitar',
    // main pane
    runnerDied: 'el runner murió a mitad del swap',
    lastSwapFailed: 'el último swap falló',
    backendResponse: 'respuesta del backend',
    dryRunOk: (m, gb) => `Dry-run OK: ${m} cabe (${gb} GB disponibles)`,
    dryRunFailed: detail => `Dry-run: ${detail}`,
    swapInProgress: 'hay un swap en curso',
    switchingToPrimary: p => `Cambiando al primario (${p})…`,
    switchingToSecondary: s => `Cambiando al secundario (${s})…`,
    slotsExchanged: 'Primario y secundario intercambiados',
    primary: 'Primario',
    secondary: 'Secundario',
    choosePlaceholder: '— elegir —',
    graceCutover: (s, target) => `corte en ${s}s → ${target}`,
    stateLabel: s => `estado: ${s}`,
    freeVram: v => `VRAM libre: ${v}`,
    engineLabel: l => `motor: ${l}`,
    serverLabel: s => `servidor: ${s}`,
    comfyOnGpu: gpus => `ComfyUI en GPU${gpus}`,
    switchAria: 'Cambiar entre el modelo primario y el secundario',
    backToPrimary: 'Volver al primario',
    switchToSecondary: 'Cambiar al secundario',
    testSwap: 'Probar swap',
    activeEngineWip: 'El motor activo está en construcción: el swap solo funciona con llama.cpp.',
    enginesHeader: 'Motores · SmartModelProvider',
    smpStartsEngine: 'SMP arranca el motor por su cuenta (sin systemd): solo necesita el motor descargado y su ruta.',
    modelsHeader: n => `Modelos (${n})`,
    addModel: 'modelo',
    runnerLog: 'Log del runner',
    noLogsYet: '(sin logs todavía)',
    // status chip
    swapFailed: 'swap falló',
    gpuJobUnloaded: 'LLM descargado por un trabajo de GPU',
    gpuJobShort: 'trabajo de GPU',
    gpuJobUnloadedNote: (job, min) => `Trabajo en curso${job ? ` (${job})` : ''}, ${min} min. Al terminar vuelve un modelo solo; no hagas swap mientras tanto.`,
    gpuJobRunningNote: (job, min) => `Trabajo de GPU en curso${job ? ` (${job})` : ''}, ${min} min. Al terminar sigue respondiendo el modelo cargado.`,
    settingsHeader: 'Ajustes',
    gpuModeTitle: 'Durante un trabajo de GPU',
    gpuModeTip: 'Qué hacer con el modelo cuando llega un trabajo de alta carga en VRAM (ComfyUI, vídeo, 3D…). Lo aplica gpu_job.sh auto.',
    gpuModeSaved: label => `Modo para trabajos de GPU: ${label}`,
    gpuModeNow: 'En este equipo, ahora:',
    automationTitle: 'Automatización',
    setAfterJob: 'Tras el trabajo responde',
    setAfterJobHint: 'El pequeño responde enseguida; el primario tarda en recargar.',
    setAfterSmall: 'Pequeño',
    setAfterPrimary: 'Primario',
    setIdle: 'Volver al primario tras inactividad',
    setIdleHint: 'Minutos sin actividad; 0 lo desactiva. Desde el próximo cambio de modelo.',
    setIdleUnit: 'min',
    setIdleBad: 'Número entero entre 0 y 720.',
    setRecover: 'Recuperación automática',
    setRecoverHint: 'Si un modelo falla al cargar o se cae, se carga otro que quepa.',
    setAutostart: 'Cargar el primario al iniciar',
    setAutostartHint: 'Al arrancar el gateway siempre se carga el primario, aunque haya quedado otro sirviendo.',
    setSaved: 'Ajuste guardado.',
    on: 'activado',
    off: 'desactivado',
    vramOf: (used, total) => `${used} / ${total}`,
    // registration
    restUnavailable: 'ctx.rest no disponible en esta version del SDK'
  },
  en: {
    // helpers
    timeJustNow: 'just now',
    timeMinutesAgo: s => `${s} min ago`,
    timeHoursAgo: n => `${n} h ago`,
    paramTemperature: 'Temperature',
    errNotANumber: 'not a number',
    errMustBeInteger: 'must be an integer',
    errOutOfRange: (min, max) => `out of range ${min}–${max}`,
    // common
    save: 'Save',
    close: 'Close',
    closeLower: 'close',
    failed: 'failed',
    // engines
    engineSaved: label => `${label} saved`,
    underConstruction: '🚧 Under construction',
    activeBadge: 'active',
    activeEngineFlash: label => `Active engine: ${label}`,
    activate: 'activate',
    port: 'Port',
    configureEngine: 'Configure engine',
    online: 'online',
    modelCount: n => `${n} model(s)`,
    noResponse: 'no response',
    binaryNotFound: 'Binary not found: set the engine path.',
    engineBinaryLabel: 'Binary (llama-server or its folder)',
    envLabel: 'Environment (KEY=value per line)',
    commonArgsLabel: 'Common args (all models)',
    test: 'Test',
    // model editor
    couldNotLoad: 'could not load',
    loading: 'Loading…',
    paramMustStartWithDash: "The parameter must start with '-' or '--'.",
    confirmDuplicateParam: key => `'${key}' is already in the list. Add it again?`,
    backendRejected: 'the backend rejected the action',
    savedWithWarnings: w => `Saved. ${w}`,
    modelSaved: n => `Model '${n}' saved`,
    confirmImportReplace: 'GGUF, mmproj, environment and parameters will be replaced. Continue?',
    importedParams: (n, notes) => `Imported ${n} parameters. ${notes}`,
    promptCopyName: n => `Name for the copy of '${n}':`,
    copyNameDefault: n => `${n}-copy`,
    confirmDeleteModel: n => `Delete '${n}' from the registry?`,
    modelDeleted: n => `Model '${n}' deleted`,
    editModelTitle: n => `Edit · ${n}`,
    newModel: 'New model',
    confirmCloseUnsaved: 'There are unsaved changes. Close?',
    name: 'Name',
    namePlaceholder: 'e.g. Qwen3.5-9B_Q5',
    nameRulesTip: 'Letters (upper/lower case), digits and . _ - + ( ) [ ] , — no spaces, / or :',
    engine: 'Engine',
    mmprojOptional: 'mmproj (optional)',
    role: 'Role',
    minVramGb: 'Min VRAM GB',
    timeoutSeconds: 'Timeout s',
    modelBinaryLabel: "Binary for this model (empty = the engine's)",
    paramsCount: n => `Parameters (${n})`,
    importLine: 'import line',
    importPlaceholder: 'llama-server -m /path/model.gguf -c 32768 --temp 0.7 …',
    import: 'Import',
    noOwnParams: "No model-specific parameters: only the engine's common args.",
    ctxSharedHint: 'The context (-c / --ctx-size) is shared: saving applies it to every model of the engine (Hermes uses a single context).',
    ctxWillApply: (ctx, names) => `On save, context ${ctx} will also be applied to: ${names}.`,
    ctxWillInherit: ctx => `No -c: on save the shared context (${ctx}) will be used.`,
    ctxSharedBadge: 'shared',
    switchValue: 'switch',
    removeParam: key => `Remove ${key}`,
    valuePlaceholder: 'value (empty = switch)',
    add: 'Add',
    searchCatalog: suffix => `Search the catalog${suffix}…`,
    noResults: 'No results.',
    validate: 'Validate',
    reset: 'Reset',
    duplicate: 'Duplicate',
    delete: 'Delete',
    binaryAcceptsLine: suffix => `The binary accepts the command line${suffix}`,
    rejected: err => `Rejected: ${err}`,
    unrecognized: list => `Unrecognized: ${list}`,
    suggestions: s => `Suggestions: ${s}`,
    apply: 'Apply',
    // model registry row
    primarySuffix: '  (primary)',
    secondarySuffix: '  (secondary)',
    editLaunchParams: 'Edit launch parameters',
    // params + presets
    modelParams: 'Model parameters',
    unsaved: 'unsaved',
    noModelsInRegistry: 'No models in the registry.',
    model: 'Model',
    loadedSuffix: n => `${n} (loaded)`,
    activePreset: 'Active preset',
    presetActive: p => `Preset '${p}' active.`,
    noPresetBase: 'No preset: base parameters.',
    basePresetOption: 'Base (no preset)',
    presetUpdated: p => `Preset '${p}' updated.`,
    baseParamsSaved: m => `Base parameters for ${m} saved.`,
    saveToPreset: p => `Save to '${p}'`,
    saveBase: 'Save base',
    discard: 'Discard',
    confirmDeletePreset: (p, m) => `Delete preset '${p}' from ${m}?`,
    presetDeleted: p => `Preset '${p}' deleted.`,
    deletePreset: 'Delete preset',
    newPresetPlaceholder: 'new preset name',
    confirmOverwritePreset: p => `Preset '${p}' already exists. Overwrite it?`,
    presetSavedActive: p => `Preset '${p}' saved and active.`,
    saveAsPreset: 'Save as preset',
    appliedNextLoad: 'Applied the next time the model is loaded',
    appliedNextLoadReload: '; to apply them to the loaded model, reload it.',
    // SmartModelProvider
    providerInstalled: 'installed',
    providerOutdated: 'outdated',
    providerFolderTaken: 'folder in use',
    providerNotInstalled: 'not installed',
    providerForeign: p => `${p} already exists and was not created by this plugin; left untouched.`,
    providerExplain: 'Exposes the registry models as a Hermes provider, so you can pick them in /model with their real context.',
    providerInstalledMsg: 'SmartModelProvider installed.',
    installProvider: 'Install SmartModelProvider',
    providerCreatesFolder: p => `Creates the folder ${p}. It does not work without this plugin.`,
    providerHowTo:
      'Pick it in /model → SmartModelProvider. Picking a model there does not load it; the list reaches Hermes when gateway or serve restarts.',
    providerUpdatedMsg: 'SmartModelProvider updated.',
    update: 'Update',
    confirmRemoveProvider: 'Remove SmartModelProvider?',
    providerRemovedMsg: 'SmartModelProvider removed.',
    remove: 'Remove',
    // main pane
    runnerDied: 'the runner died mid-swap',
    lastSwapFailed: 'the last swap failed',
    backendResponse: 'backend response',
    dryRunOk: (m, gb) => `Dry-run OK: ${m} fits (${gb} GB available)`,
    dryRunFailed: detail => `Dry-run: ${detail}`,
    swapInProgress: 'a swap is in progress',
    switchingToPrimary: p => `Switching to primary (${p})…`,
    switchingToSecondary: s => `Switching to secondary (${s})…`,
    slotsExchanged: 'Primary and secondary swapped',
    primary: 'Primary',
    secondary: 'Secondary',
    choosePlaceholder: '— choose —',
    graceCutover: (s, target) => `cutover in ${s}s → ${target}`,
    stateLabel: s => `state: ${s}`,
    freeVram: v => `Free VRAM: ${v}`,
    engineLabel: l => `engine: ${l}`,
    serverLabel: s => `server: ${s}`,
    comfyOnGpu: gpus => `ComfyUI on GPU${gpus}`,
    switchAria: 'Switch between the primary and secondary model',
    backToPrimary: 'Back to primary',
    switchToSecondary: 'Switch to secondary',
    testSwap: 'Test swap',
    activeEngineWip: 'The active engine is under construction: swap only works with llama.cpp.',
    enginesHeader: 'Engines · SmartModelProvider',
    smpStartsEngine: 'SMP starts the engine on its own (no systemd): it only needs the engine downloaded and its path.',
    modelsHeader: n => `Models (${n})`,
    addModel: 'model',
    runnerLog: 'Runner log',
    noLogsYet: '(no logs yet)',
    // status chip
    swapFailed: 'swap failed',
    gpuJobUnloaded: 'LLM unloaded for a GPU job',
    gpuJobShort: 'GPU job',
    gpuJobUnloadedNote: (job, min) => `Job running${job ? ` (${job})` : ''}, ${min} min. A model comes back by itself when it ends; do not swap meanwhile.`,
    gpuJobRunningNote: (job, min) => `GPU job running${job ? ` (${job})` : ''}, ${min} min. The loaded model keeps answering when it ends.`,
    settingsHeader: 'Settings',
    gpuModeTitle: 'During a GPU job',
    gpuModeTip: 'What to do with the model when a VRAM-heavy job arrives (ComfyUI, video, 3D…). Applied by gpu_job.sh auto.',
    gpuModeSaved: label => `GPU job mode: ${label}`,
    gpuModeNow: 'On this machine, now:',
    automationTitle: 'Automation',
    setAfterJob: 'After the job, answer with',
    setAfterJobHint: 'The small one answers right away; the primary takes time to reload.',
    setAfterSmall: 'Small',
    setAfterPrimary: 'Primary',
    setIdle: 'Back to the primary when idle',
    setIdleHint: 'Idle minutes; 0 disables it. From the next model change.',
    setIdleUnit: 'min',
    setIdleBad: 'Whole number between 0 and 720.',
    setRecover: 'Auto-recover',
    setRecoverHint: 'If a model fails to load or dies, another one that fits is loaded.',
    setAutostart: 'Load the primary at start-up',
    setAutostartHint: 'When the gateway starts the primary is always loaded, even if another model was left serving.',
    setSaved: 'Setting saved.',
    on: 'on',
    off: 'off',
    vramOf: (used, total) => `${used} / ${total}`,
    // registration
    restUnavailable: 'ctx.rest is not available in this SDK version'
  }
}

let LANG = 'es'
const langOf = loc => (/^es([-_]|$)/i.test(String(loc || '')) ? 'es' : 'en')
function t(key, ...args) {
  const d = I18N[LANG] || I18N.es
  let v = d[key]
  if (v === undefined) v = I18N.es[key]
  if (v === undefined) return key
  return typeof v === 'function' ? v(...args) : v
}
const withLang = path => path + (path.includes('?') ? '&' : '?') + 'lang=' + LANG

const ID = 'smart-model-swap'
const API = '/api/plugins/smart-model-swap'

// --------------------------------------------------------------------------- helpers

function stateTone(state) {
  if (isBusy(state)) return 'warn'
  if (state === 'failed') return 'bad'
  if (state === 'idle') return 'muted'
  return 'good'
}

// grace/swapping/restoring all mean "do not touch" (M8)
function isBusy(state) {
  return state === 'swapping' || state === 'grace' || state === 'restoring' || state === 'starting'
}

function freeGb(v) {
  if (v == null) return '?'
  const g = v / 1024
  return g >= 1 ? `${g.toFixed(1)} GB` : `${Math.max(0, Math.round(v))} MB`
}

function timeAgo(ts) {
  if (!ts) return ''
  // accept epoch-seconds or ISO-8601 (state.json stores ISO strings)
  const secs =
    typeof ts === 'number' ? ts : Date.parse(ts) / 1000
  if (!isFinite(secs)) return ''
  const s = Math.max(0, Math.floor((Date.now() / 1000 - secs) / 60))
  if (s < 1) return t('timeJustNow')
  if (s < 60) return t('timeMinutesAgo', s)
  const h = Math.floor(s / 60)
  return t('timeHoursAgo', h)
}

const PARAM_LABELS = {
  temperature: () => t('paramTemperature'),
  top_p: 'Top P',
  top_k: 'Top K',
  min_p: 'Min P',
  presence_penalty: 'Presence',
  repeat_penalty: 'Repeat'
}

function paramError(spec, raw) {
  const text = String(raw == null ? '' : raw).trim().replace(',', '.')
  const n = Number(text)
  if (text === '' || !isFinite(n)) return t('errNotANumber')
  if (spec.integer && Math.floor(n) !== n) return t('errMustBeInteger')
  if (n < spec.min || n > spec.max) return t('errOutOfRange', spec.min, spec.max)
  return null
}

// --------------------------------------------------------------------------- tiny element helper

// h(type, props, ...children) over the automatic JSX runtime (key lifted out of props).
function h(type, props, ...children) {
  const { key, ...rest } = props || {}
  const kids = children.filter(c => c !== undefined)
  if (kids.length > 1) return jsxs(type, { ...rest, children: kids }, key)
  return jsx(type, kids.length ? { ...rest, children: kids[0] } : rest, key)
}

// The Desktop app ships precompiled CSS: a Tailwind class the app itself never
// uses does nothing in a plugin. Anything outside its set goes inline.
const WARN = { color: 'var(--ui-warning, #d29922)' }
const BAD_BORDER = { borderColor: 'var(--dt-destructive, #e5484d)' }
const RULE = '1px solid var(--ui-stroke-secondary)'

const LABEL = 'text-[10px] text-(--ui-text-tertiary)'
const MONO = 'font-mono text-[11px]'

// Same quoting the backend tokenizer (smp/cmdline.py) understands.
function quoteArg(s) {
  s = String(s)
  if (s !== '' && /^[A-Za-z0-9_\-.,:/=+@%]+$/.test(s)) return s
  return '"' + s.replace(/"/g, '\\"') + '"'
}
const joinArgs = list => (list || []).map(quoteArg).join(' ')
const envToText = env =>
  Object.keys(env || {})
    .map(k => `${k}=${env[k]}`)
    .join('\n')
let rowSeq = 0
function mkRow(key, value) {
  rowSeq += 1
  return { id: `r${rowSeq}`, key, value }
}
const withIds = rows => (rows || []).map(r => mkRow(r.key, r.value == null ? null : String(r.value)))
const plainRows = rows => rows.map(r => ({ key: r.key.trim(), value: r.value == null ? null : String(r.value) }))

function Field({ label, children }) {
  return h('label', { className: 'flex min-w-0 flex-col gap-0.5' }, h('span', { className: LABEL }, label), children)
}

// --------------------------------------------------------------------------- engines (SmartModelProvider)

function EngineRow({ e, health, onCall, flash }) {
  const wip = e.status === 'wip'
  const [f, setF] = useState(null)
  const [pending, setPending] = useState(false)
  const [version, setVersion] = useState(null)
  const up = !!(health && health.up)
  const n = (health && health.models && health.models.length) || 0
  const set = k => ev => {
    const v = ev.target.value
    setF(o => ({ ...o, [k]: v }))
  }

  function open() {
    setF({
      bin: e.bin || '',
      host: e.host || '127.0.0.1',
      port: String(e.port || ''),
      env: envToText(e.env),
      common: joinArgs(e.common_args)
    })
  }
  function save() {
    const body = { engine: e.id, host: f.host.trim(), port: parseInt(f.port, 10) || null }
    if (!wip) Object.assign(body, { bin: f.bin.trim(), env: f.env, common_args: f.common })
    setPending(true)
    onCall('/engines/update', body)
      .then(r => {
        if (r && r.ok) {
          setF(null)
          const w = r.warnings || []
          flash(w.length ? 'warn' : 'ok', w.length ? w.join(' · ') : t('engineSaved', e.label))
        }
      })
      .finally(() => setPending(false))
  }
  function test() {
    setPending(true)
    setVersion(null)
    onCall('/engines/test', { engine: e.id, bin: f ? f.bin.trim() : '' }, null, true)
      .then(r => setVersion(r && r.ok ? { ok: true, text: r.version } : { ok: false, text: (r && r.error) || t('failed') }))
      .finally(() => setPending(false))
  }

  return h(
    'div',
    {
      className: cn(
        'flex flex-col gap-1 rounded-[4px] border px-2 py-1.5 text-xs',
        e.active ? 'border-(--ui-accent) bg-(--ui-accent)/10' : 'border-(--ui-stroke-secondary)',
        wip && 'border-dashed opacity-70'
      )
    },
    h(
      'div',
      { className: 'flex items-center gap-1.5' },
      h(StatusDot, { tone: up ? 'good' : wip ? 'muted' : 'bad', className: 'size-2' }),
      h('span', { className: 'min-w-0 flex-1 truncate font-medium' }, e.label),
      wip
        ? h(Badge, { variant: 'muted', style: WARN, className: 'px-1 text-[10px]' }, t('underConstruction'))
        : e.active
          ? h(Badge, { variant: 'muted', className: 'px-1 text-[10px]' }, t('activeBadge'))
          : h(
              'button',
              {
                type: 'button',
                className: 'text-[10px] text-(--ui-text-tertiary) underline hover:text-foreground',
                onClick: () => onCall('/engines/select', { engine: e.id }).then(r => r && r.ok && flash('ok', t('activeEngineFlash', e.label)))
              },
              t('activate')
            ),
      h(
        'button',
        {
          type: 'button',
          title: wip ? t('port') : t('configureEngine'),
          onClick: () => (f ? setF(null) : open()),
          className: 'rounded p-0.5 text-(--ui-text-tertiary) hover:bg-(--chrome-action-hover) hover:text-foreground'
        },
        jsx(Codicon, { name: 'gear', size: 13 })
      )
    ),
    h(
      'div',
      { className: 'truncate text-[10px] text-(--ui-text-tertiary)', title: e.bin || '' },
      `${e.host}:${e.port}${up ? ` · ${t('online')}${n ? ` · ${t('modelCount', n)}` : ''}` : ` · ${t('noResponse')}`}`
    ),
    !wip && e.bin_ok === false
      ? h('div', { className: 'text-[10px] text-destructive' }, t('binaryNotFound'))
      : null,
    f
      ? h(
          'div',
          { className: 'flex flex-col gap-1.5 pt-1' },
          !wip
            ? h(
                Field,
                { label: t('engineBinaryLabel') },
                h('input', { className: cn(FIELD, MONO), value: f.bin, onChange: set('bin') })
              )
            : null,
          h(
            'div',
            { className: 'flex gap-1.5' },
            h(Field, { label: 'Host' }, h('input', { className: FIELD, value: f.host, onChange: set('host') })),
            h(
              Field,
              { label: t('port') },
              h('input', {
                className: FIELD,
                value: f.port,
                onChange: ev => {
                  const v = ev.target.value.replace(/[^0-9]/g, '')
                  setF(o => ({ ...o, port: v }))
                }
              })
            )
          ),
          !wip
            ? h(
                Field,
                { label: t('envLabel') },
                h('textarea', { rows: 2, className: cn(FIELD, MONO), value: f.env, onChange: set('env'), placeholder: 'CUDA_VISIBLE_DEVICES=0,1' })
              )
            : null,
          !wip
            ? h(
                Field,
                { label: t('commonArgsLabel') },
                h('textarea', { rows: 3, className: cn(FIELD, MONO), value: f.common, onChange: set('common') })
              )
            : null,
          version ? h('div', { className: cn('text-[10px] break-all', version.ok ? 'text-(--ui-accent)' : 'text-destructive') }, version.text) : null,
          h(
            'div',
            { className: 'flex gap-1.5' },
            h(Button, { size: 'sm', disabled: pending, onClick: save }, t('save')),
            !wip ? h(Button, { variant: 'outline', size: 'sm', disabled: pending, onClick: test }, t('test')) : null,
            h(Button, { variant: 'ghost', size: 'sm', onClick: () => setF(null) }, '×')
          )
        )
      : null
  )
}

// --------------------------------------------------------------------------- model editor (Launcher-style)

const EMPTY_DRAFT = {
  name: '', original_name: null, engine: 'llamacpp', gguf: '', mmproj: '', bin: '', env: '',
  role: '', description: '', min_free_vram_gb: '8', load_timeout_s: '240'
}
const draftFrom = m => ({
  name: m.name, original_name: m.name, engine: m.engine || 'llamacpp', gguf: m.gguf || '', mmproj: m.mmproj || '',
  bin: m.bin || '', env: envToText(m.env), role: m.role || '', description: m.description || '',
  min_free_vram_gb: String(m.min_free_vram_gb ?? ''), load_timeout_s: String(m.load_timeout_s ?? '')
})
const bodyOf = (d, rows) => ({
  name: d.name.trim(), original_name: d.original_name, engine: d.engine, gguf: d.gguf.trim(), mmproj: d.mmproj.trim(),
  bin: d.bin.trim(), env: d.env, rows: plainRows(rows), role: d.role.trim(), description: d.description.trim(),
  min_free_vram_gb: parseFloat(String(d.min_free_vram_gb).replace(',', '.')) || 0,
  load_timeout_s: parseInt(d.load_timeout_s, 10) || 240
})
const sigOf = (d, rows) => JSON.stringify(bodyOf(d, rows))

function ModelEditor({ name, engines, primary, models, rest, onCall, onClose, onSelect, flash }) {
  const [d, setD] = useState(null)
  const [rows, setRows] = useState([])
  const [base, setBase] = useState('')
  const [catalog, setCatalog] = useState(null)
  const [preview, setPreview] = useState(null)
  const [pending, setPending] = useState(false)
  const [msg, setMsg] = useState(null)
  const [q, setQ] = useState('')
  const [manual, setManual] = useState({ key: '', value: '' })
  const [importText, setImportText] = useState(null)

  function load(n) {
    setMsg(null)
    setPreview(null)
    if (!n) {
      const dd = { ...EMPTY_DRAFT }
      setD(dd)
      setRows([])
      setBase(sigOf(dd, []))
      return
    }
    rest(`/models/detail?name=${encodeURIComponent(n)}`)
      .then(r => {
        if (!r || r.ok === false) return setMsg({ kind: 'bad', text: (r && r.error) || t('couldNotLoad') })
        const dd = draftFrom(r.model)
        const rr = withIds(r.rows)
        setD(dd)
        setRows(rr)
        setBase(sigOf(dd, rr))
        setPreview(r)
      })
      .catch(e => setMsg({ kind: 'bad', text: String((e && e.message) || e) }))
  }
  useEffect(() => load(name), [name])
  const engineId = d ? d.engine : 'llamacpp'
  useEffect(() => {
    rest(`/params/catalog?engine=${encodeURIComponent(engineId)}${name ? `&model=${encodeURIComponent(name)}` : ''}`)
      .then(c => setCatalog(c && c.ok ? c : null))
      .catch(() => setCatalog(null))
  }, [name, engineId])

  if (!d) return h('div', { className: LABEL }, msg ? msg.text : t('loading'))

  const dirty = sigOf(d, rows) !== base
  const byPrefix = {}
  ;((catalog && catalog.groups) || []).forEach(g => (g.parameters || []).forEach(p => (byPrefix[p.prefix] = p)))
  const have = {}
  rows.forEach(r => (have[r.key] = true))
  const set = k => e => {
    const v = e.target.value
    setD(o => ({ ...o, [k]: v }))
  }
  const setRow = (id, patch) => setRows(rs => rs.map(r => (r.id === id ? { ...r, ...patch } : r)))
  const removeRow = id => setRows(rs => rs.filter(r => r.id !== id))

  function addParam(key, value) {
    key = String(key || '').trim()
    if (!/^-/.test(key)) {
      setMsg({ kind: 'bad', text: t('paramMustStartWithDash') })
      return false
    }
    if (have[key] && !window.confirm(t('confirmDuplicateParam', key))) return false
    setRows(rs => rs.concat([mkRow(key, value)]))
    setMsg(null)
    return true
  }
  function run(path, body, after) {
    setPending(true)
    setMsg(null)
    return onCall(path, body, null, true)
      .then(r => {
        if (!r || r.ok === false) {
          setMsg({ kind: 'bad', text: (r && r.error) || t('backendRejected') })
          return r
        }
        return after(r)
      })
      .finally(() => setPending(false))
  }
  const save = () =>
    run('/models', bodyOf(d, rows), r => {
      const w = r.warnings || []
      const notes = r.notes || []
      flash(
        w.length ? 'warn' : 'ok',
        w.length
          ? t('savedWithWarnings', [...w, ...notes].join(' · '))
          : t('modelSaved', r.name) + (notes.length ? ` · ${notes.join(' · ')}` : '')
      )
      if (name !== r.name) onSelect(r.name)
      else load(r.name)
    })
  const doPreview = () => run('/models/preview', bodyOf(d, rows), r => setPreview(r))
  const doImport = () =>
    run('/models/parse_cmd', { cmdline: importText || '' }, r => {
      if (rows.length && !window.confirm(t('confirmImportReplace'))) return
      setD(o => ({
        ...o,
        gguf: r.gguf || o.gguf,
        mmproj: r.mmproj || '',
        bin: r.bin || o.bin,
        env: Object.keys(r.env || {}).length ? envToText(r.env) : o.env
      }))
      setRows(withIds(r.rows))
      setImportText(null)
      setMsg({ kind: 'ok', text: t('importedParams', r.rows.length, (r.notes || []).join(' · ')) })
    })
  const doDuplicate = () => {
    const n = window.prompt(t('promptCopyName', name), t('copyNameDefault', name))
    if (n) run('/models/duplicate', { name, new_name: n.trim() }, r => onSelect(r.name))
  }
  const doDelete = () => {
    if (window.confirm(t('confirmDeleteModel', name)))
      run('/models/delete', { name }, () => {
        flash('ok', t('modelDeleted', name))
        onClose()
      })
  }

  const needle = q.trim().toLowerCase()
  const hits = []
  if (needle)
    ((catalog && catalog.groups) || []).forEach(g =>
      (g.parameters || []).forEach(p => {
        if (hits.length < 40 && `${p.prefix} ${p.name} ${p.description || ''}`.toLowerCase().includes(needle)) hits.push(p)
      })
    )
  const pv = preview
  // Shared context: what saving this draft does to the other models' -c.
  const ctxEffect = (() => {
    const engine = d.engine || 'llamacpp'
    const others = (models || []).filter(m => m.name !== name && m.name !== d.name && (m.engine || 'llamacpp') === engine)
    const ctxRow = rows.find(r => r.key === '-c' || r.key === '--ctx-size')
    const draftCtx = ctxRow ? String(ctxRow.value ?? '').trim() : ''
    if (draftCtx) {
      const hit = others.filter(m => String(m.ctx ?? '') !== draftCtx).map(m => m.name)
      return hit.length ? t('ctxWillApply', draftCtx, hit.join(', ')) : null
    }
    const shared = (others.find(m => m.name === primary && m.ctx) || others.find(m => m.ctx) || {}).ctx
    return shared ? t('ctxWillInherit', shared) : null
  })()

  return h(
    'div',
    {
      className: 'flex flex-col gap-1.5 rounded-[4px] border p-2 text-xs',
      style: { borderColor: 'color-mix(in srgb, var(--ui-accent) 60%, transparent)' }
    },
    h(
      'div',
      { className: 'flex items-center gap-1.5' },
      h('span', { className: 'flex-1 font-medium' }, (name ? t('editModelTitle', name) : t('newModel')) + (dirty ? ' ●' : '')),
      h(
        'button',
        {
          type: 'button',
          title: t('close'),
          className: 'rounded p-0.5 text-(--ui-text-tertiary) hover:text-foreground',
          onClick: () => (!dirty || window.confirm(t('confirmCloseUnsaved'))) && onClose()
        },
        jsx(Codicon, { name: 'close', size: 12 })
      )
    ),
    h(
      'div',
      { className: 'flex gap-1.5' },
      h(Field, { label: t('name') }, h('input', { className: FIELD, value: d.name, onChange: set('name'), maxLength: 64, placeholder: t('namePlaceholder'), title: t('nameRulesTip') })),
      h(
        Field,
        { label: t('engine') },
        h(
          'select',
          { className: FIELD, value: d.engine, onChange: set('engine') },
          (engines || []).map(e =>
            jsx('option', { value: e.id, disabled: e.status === 'wip', children: e.label + (e.status === 'wip' ? ' 🚧' : '') }, e.id)
          )
        )
      )
    ),
    h(Field, { label: 'GGUF' }, h('input', { className: cn(FIELD, MONO), value: d.gguf, onChange: set('gguf') })),
    h(Field, { label: t('mmprojOptional') }, h('input', { className: cn(FIELD, MONO), value: d.mmproj, onChange: set('mmproj') })),
    h(
      'div',
      { className: 'flex gap-1.5' },
      h(Field, { label: t('role') }, h('input', { className: FIELD, value: d.role, onChange: set('role') })),
      h(Field, { label: t('minVramGb') }, h('input', { className: FIELD, value: d.min_free_vram_gb, onChange: set('min_free_vram_gb') })),
      h(Field, { label: t('timeoutSeconds') }, h('input', { className: FIELD, value: d.load_timeout_s, onChange: set('load_timeout_s') }))
    ),
    h(
      Field,
      { label: t('modelBinaryLabel') },
      h('input', { className: cn(FIELD, MONO), value: d.bin, onChange: set('bin') })
    ),
    h(
      Field,
      { label: t('envLabel') },
      h('textarea', { rows: 2, className: cn(FIELD, MONO), value: d.env, onChange: set('env'), placeholder: 'MTMD_BACKEND_DEVICE=cuda1' })
    ),
    h(
      'div',
      { className: 'flex items-center gap-1.5 pt-1' },
      h('span', { className: 'flex-1 text-[11px] font-medium' }, t('paramsCount', rows.length)),
      h(
        'button',
        {
          type: 'button',
          className: 'text-[10px] text-(--ui-text-tertiary) underline hover:text-foreground',
          onClick: () => setImportText(importText == null ? '' : null)
        },
        importText == null ? t('importLine') : t('closeLower')
      )
    ),
    importText != null
      ? h(
          'div',
          { className: 'flex flex-col gap-1' },
          h('textarea', {
            rows: 3,
            className: cn(FIELD, MONO),
            value: importText,
            onChange: e => setImportText(e.target.value),
            placeholder: t('importPlaceholder')
          }),
          h('div', null, h(Button, { size: 'sm', disabled: pending || !importText.trim(), onClick: doImport }, t('import')))
        )
      : null,
    h(
      'div',
      { className: 'flex max-h-72 flex-col gap-1 overflow-y-auto' },
      rows.length === 0 ? h('div', { className: LABEL }, t('noOwnParams')) : null,
      h('div', { className: LABEL }, t('ctxSharedHint')),
      ctxEffect ? h('div', { style: WARN, className: 'text-[10px]' }, ctxEffect) : null,
      rows.map(r => {
        const info = byPrefix[r.key]
        const bad = pv && (pv.unsupported || []).includes(r.key)
        let valueEl
        if (r.value == null) valueEl = h('span', { className: cn(LABEL, 'flex-1') }, t('switchValue'))
        else if (info && info.type === 'select' && (info.options || []).length) {
          const opts = info.options.includes(r.value) || r.value === '' ? info.options : [r.value, ...info.options]
          valueEl = h(
            'select',
            { className: cn(FIELD, 'flex-1'), value: r.value, onChange: e => setRow(r.id, { value: e.target.value }) },
            opts.map(o => jsx('option', { value: o, children: o }, o))
          )
        } else
          valueEl = h('input', {
            className: cn(FIELD, MONO, 'flex-1'),
            value: r.value,
            onChange: e => setRow(r.id, { value: e.target.value })
          })
        return h(
          'div',
          { key: r.id, className: 'flex items-center gap-1', title: info ? `${info.name} — ${info.description || ''}` : '' },
          h('input', {
            className: cn(FIELD, MONO, 'shrink-0'),
            style: { width: '38%', ...(bad ? BAD_BORDER : null) },
            value: r.key,
            onChange: e => setRow(r.id, { key: e.target.value })
          }),
          valueEl,
          r.key === '-c' || r.key === '--ctx-size'
            ? h('span', { className: cn(LABEL, 'shrink-0'), title: t('ctxSharedHint') }, t('ctxSharedBadge'))
            : null,
          h(
            'button',
            { type: 'button', title: t('removeParam', r.key), className: 'shrink-0 rounded p-0.5 text-(--ui-text-tertiary) hover:text-destructive', onClick: () => removeRow(r.id) },
            jsx(Codicon, { name: 'close', size: 11 })
          )
        )
      })
    ),
    h(
      'div',
      { className: 'flex items-center gap-1' },
      h('input', {
        className: cn(FIELD, MONO, 'shrink-0'),
        style: { width: '38%' },
        placeholder: '--flag',
        value: manual.key,
        onChange: e => setManual({ ...manual, key: e.target.value })
      }),
      h('input', {
        className: cn(FIELD, MONO, 'flex-1'),
        placeholder: t('valuePlaceholder'),
        value: manual.value,
        onChange: e => setManual({ ...manual, value: e.target.value })
      }),
      h(
        'button',
        {
          type: 'button',
          title: t('add'),
          disabled: !manual.key.trim(),
          className: 'shrink-0 rounded p-0.5 text-(--ui-text-tertiary) hover:text-foreground disabled:opacity-40',
          onClick: () => addParam(manual.key, manual.value.trim() === '' ? null : manual.value.trim()) && setManual({ key: '', value: '' })
        },
        jsx(Codicon, { name: 'add', size: 12 })
      )
    ),
    h('input', {
      className: FIELD,
      placeholder: t('searchCatalog', catalog ? ` (${catalog.count})` : ''),
      value: q,
      onChange: e => setQ(e.target.value)
    }),
    hits.length
      ? h(
          'div',
          { className: 'flex max-h-56 flex-col gap-0.5 overflow-y-auto rounded-[3px] border border-(--ui-stroke-secondary) p-1' },
          hits.map(p =>
            h(
              'button',
              {
                key: p.prefix,
                type: 'button',
                title: p.description || '',
                className: cn(
                  'flex flex-col items-start rounded px-1 py-0.5 text-left hover:bg-(--chrome-action-hover)',
                  (p.status === 'removed' || p.status === 'unknown') && 'opacity-50'
                ),
                onClick: () =>
                  addParam(
                    p.prefix,
                    p.type === 'checkbox' ? null : p.type === 'select' ? String(p.default || (p.options || [])[0] || '') : String(p.default ?? '')
                  )
              },
              h('span', { className: MONO }, `${p.prefix}${have[p.prefix] ? ' ✓' : ''}`),
              h('span', { className: cn(LABEL, 'line-clamp-2') }, `${p.name} — ${p.description || ''}`)
            )
          )
        )
      : needle
        ? h('div', { className: LABEL }, t('noResults'))
        : null,
    h(
      'div',
      { className: 'flex flex-wrap items-center gap-1.5 pt-1' },
      h(Button, { size: 'sm', disabled: pending || !dirty, onClick: save }, t('save')),
      h(Button, { variant: 'outline', size: 'sm', disabled: pending, onClick: doPreview }, t('validate')),
      h(Button, { variant: 'ghost', size: 'sm', disabled: pending || !dirty, onClick: () => load(name) }, t('reset')),
      name ? h(Button, { variant: 'ghost', size: 'sm', disabled: pending, onClick: doDuplicate }, t('duplicate')) : null,
      name && name !== primary
        ? h(Button, { variant: 'ghost', size: 'sm', className: 'text-destructive', disabled: pending, onClick: doDelete }, t('delete'))
        : null
    ),
    msg ? h('div', { className: cn('text-[10px]', msg.kind === 'bad' ? 'text-destructive' : 'text-(--ui-accent)') }, msg.text) : null,
    pv && pv.ok !== false
      ? h(
          'div',
          { className: 'flex flex-col gap-1' },
          h(
            'div',
            { className: 'flex items-center gap-1.5 text-[10px]' },
            h(StatusDot, { tone: pv.validate && pv.validate.ok ? 'good' : 'bad', className: 'size-2' }),
            h(
              'span',
              { className: 'text-(--ui-text-tertiary)' },
              pv.validate && pv.validate.ok ? t('binaryAcceptsLine', pv.build ? ` (${pv.build})` : '') : t('rejected', (pv.validate && pv.validate.error) || '?')
            )
          ),
          (pv.unsupported || []).length
            ? h('div', { style: WARN, className: 'text-[10px]' }, t('unrecognized', pv.unsupported.join(', ')))
            : null,
          (pv.fix_notes || []).length
            ? h(
                'div',
                { style: WARN, className: 'flex flex-col gap-1 text-[10px]' },
                t('suggestions', pv.fix_notes.join(' · ')),
                pv.fixed_rows
                  ? h('div', null, h(Button, { variant: 'outline', size: 'sm', onClick: () => setRows(withIds(pv.fixed_rows)) }, t('apply')))
                  : null
              )
            : null,
          h(
            'pre',
            { className: 'max-h-32 overflow-y-auto rounded-[3px] border border-(--ui-stroke-secondary) p-1.5 font-mono text-[10px] break-all whitespace-pre-wrap' },
            pv.cmdline || ''
          )
        )
      : null
  )
}

// --------------------------------------------------------------------------- model registry row

function ModelRow({ m, isPrimary, isSecondary, editing, onEdit }) {
  return h(
    'div',
    {
      className: cn(
        'flex items-center gap-2 rounded-[3px] px-1 py-1 text-xs hover:bg-(--chrome-action-hover)',
        editing && 'bg-(--ui-accent)/10'
      )
    },
    h(StatusDot, { tone: m.exists ? 'good' : 'warn', className: 'size-2' }),
    h('span', { className: 'min-w-0 flex-1 truncate font-medium', title: m.gguf }, m.name + (isPrimary ? t('primarySuffix') : isSecondary ? t('secondarySuffix') : '')),
    m.role ? h(Badge, { variant: 'muted', className: 'px-1 text-[10px]' }, m.role) : null,
    m.min_free_vram_gb ? h('span', { className: 'shrink-0 text-(--ui-text-tertiary)' }, `${m.min_free_vram_gb} GB`) : null,
    h(
      'button',
      { type: 'button', title: t('editLaunchParams'), onClick: () => onEdit(m.name), className: 'shrink-0 rounded p-0.5 text-(--ui-text-tertiary) hover:text-foreground' },
      jsx(Codicon, { name: 'edit', size: 12 })
    )
  )
}

// --------------------------------------------------------------------------- model parameters + presets

const FIELD =
  'w-full min-w-0 rounded-[3px] border border-(--ui-stroke-secondary) bg-transparent px-1.5 py-1 text-xs outline-none focus:border-(--ui-accent)'

// Sampling params of one model + its presets. Values are applied when the
// model is next loaded: base args, overridden by the active preset.
function ParamsEditor({ models, specs, current, primary, onCall }) {
  const [open, setOpen] = useState(false)
  const [picked, setPicked] = useState('')
  const [edit, setEdit] = useState(null) // {key, vals}: unsaved edits for one model+preset
  const [name, setName] = useState('')
  const [pending, setPending] = useState(false)
  const [note, setNote] = useState(null)

  if (!open) {
    return jsx(Button, {
      variant: 'outline',
      size: 'sm',
      onClick: () => setOpen(true),
      children: jsxs('span', {
        children: [jsx(Codicon, { name: 'settings', size: 12 }), ' ' + t('modelParams')]
      })
    })
  }

  const model =
    picked && models.some(m => m.name === picked)
      ? picked
      : models.some(m => m.name === current)
        ? current
        : primary || (models[0] && models[0].name) || ''
  const entry = models.find(m => m.name === model) || null
  const preset = (entry && entry.active_preset) || ''
  const key = `${model}|${preset}`
  const saved = (entry && entry.params) || {}
  const vals = edit && edit.key === key ? edit.vals : saved
  const dirty = specs.some(sp => String(vals[sp.key]) !== String(saved[sp.key]))
  const invalid = specs.some(sp => paramError(sp, vals[sp.key]))
  const presetNames = Object.keys((entry && entry.presets) || {})
  const newName = name.trim()

  function send(path, body, okText) {
    setPending(true)
    setNote(null)
    return onCall(path, body)
      .then(r => {
        if (r && r.ok) {
          setEdit(null)
          setNote(okText)
        }
        return r
      })
      .finally(() => setPending(false))
  }

  return jsxs('div', {
    className: 'flex flex-col gap-1.5 rounded-[4px] border border-(--ui-stroke-secondary) p-2 text-xs',
    children: [
      jsxs('div', {
        className: 'flex items-center gap-1.5',
        children: [
          jsx(Codicon, { name: 'settings', size: 13 }),
          jsx('span', { className: 'font-medium', children: t('modelParams') }),
          pending
            ? jsx(GlyphSpinner, { className: 'ml-auto text-[10px]' })
            : dirty
              ? jsx('span', { className: 'ml-auto text-[10px] text-(--ui-text-tertiary)', children: t('unsaved') })
              : null
        ]
      }),
      !entry
        ? jsx('div', { className: 'text-(--ui-text-tertiary)', children: t('noModelsInRegistry') })
        : jsxs('div', {
            className: 'flex flex-col gap-1.5',
            children: [
              jsxs('div', {
                className: 'grid grid-cols-2 gap-1.5',
                children: [
                  jsxs('label', {
                    className: 'flex min-w-0 flex-col gap-0.5',
                    children: [
                      jsx('span', { className: 'text-[10px] text-(--ui-text-tertiary)', children: t('model') }),
                      jsx('select', {
                        value: model,
                        disabled: pending,
                        onChange: e => {
                          setPicked(e.target.value)
                          setEdit(null)
                          setNote(null)
                        },
                        className: FIELD,
                        children: models.map(m =>
                          jsx(
                            'option',
                            { value: m.name, children: m.name === current ? t('loadedSuffix', m.name) : m.name },
                            m.name
                          )
                        )
                      })
                    ]
                  }),
                  jsxs('label', {
                    className: 'flex min-w-0 flex-col gap-0.5',
                    children: [
                      jsx('span', { className: 'text-[10px] text-(--ui-text-tertiary)', children: t('activePreset') }),
                      jsxs('select', {
                        value: preset,
                        disabled: pending,
                        onChange: e => {
                          const p = e.target.value
                          send(
                            '/params/preset/select',
                            { model, preset: p },
                            p ? t('presetActive', p) : t('noPresetBase')
                          )
                        },
                        className: FIELD,
                        children: [
                          jsx('option', { value: '', children: t('basePresetOption') }, '__base'),
                          presetNames.map(p => jsx('option', { value: p, children: p }, p))
                        ]
                      })
                    ]
                  })
                ]
              }),
              jsx('div', {
                className: 'grid grid-cols-2 gap-1.5',
                children: specs.map(sp => {
                  const err = paramError(sp, vals[sp.key])
                  return jsxs(
                    'label',
                    {
                      className: 'flex min-w-0 flex-col gap-0.5',
                      title: `${sp.flag} · ${sp.min}–${sp.max}`,
                      children: [
                        jsx('span', {
                          className: 'text-[10px] text-(--ui-text-tertiary)',
                          children: (typeof PARAM_LABELS[sp.key] === 'function' ? PARAM_LABELS[sp.key]() : PARAM_LABELS[sp.key]) || sp.key
                        }),
                        jsx('input', {
                          type: 'number',
                          inputMode: 'decimal',
                          min: sp.min,
                          max: sp.max,
                          step: sp.integer ? 1 : 0.05,
                          value: vals[sp.key] == null ? '' : vals[sp.key],
                          disabled: pending,
                          'aria-invalid': err ? 'true' : 'false',
                          onChange: e => {
                            setEdit({ key, vals: { ...vals, [sp.key]: e.target.value } })
                            setNote(null)
                          },
                          className: cn(FIELD, 'font-mono text-[11px]'),
                          style: err ? BAD_BORDER : undefined
                        }),
                        err ? jsx('span', { className: 'text-[10px] text-destructive', children: err }) : null
                      ]
                    },
                    sp.key
                  )
                })
              }),
              jsxs('div', {
                className: 'flex flex-wrap items-center gap-1.5',
                children: [
                  jsx(Button, {
                    size: 'sm',
                    disabled: pending || !dirty || invalid,
                    onClick: () =>
                      send(
                        '/params/save',
                        { model, params: vals, preset },
                        preset ? t('presetUpdated', preset) : t('baseParamsSaved', model)
                      ),
                    children: preset ? t('saveToPreset', preset) : t('saveBase')
                  }),
                  dirty
                    ? jsx(Button, {
                        variant: 'ghost',
                        size: 'sm',
                        disabled: pending,
                        onClick: () => {
                          setEdit(null)
                          setNote(null)
                        },
                        children: t('discard')
                      })
                    : null,
                  preset
                    ? jsx(Button, {
                        variant: 'ghost',
                        size: 'sm',
                        className: 'text-destructive',
                        disabled: pending,
                        onClick: () => {
                          if (window.confirm(t('confirmDeletePreset', preset, model)))
                            send('/params/preset/delete', { model, preset }, t('presetDeleted', preset))
                        },
                        children: t('deletePreset')
                      })
                    : null
                ]
              }),
              jsxs('div', {
                className: 'flex items-center gap-1.5',
                children: [
                  jsx('input', {
                    value: name,
                    maxLength: 40,
                    disabled: pending,
                    onChange: e => setName(e.target.value),
                    placeholder: t('newPresetPlaceholder'),
                    className: cn(FIELD, 'flex-1')
                  }),
                  jsx(Button, {
                    size: 'sm',
                    variant: 'outline',
                    disabled: pending || !newName || invalid,
                    onClick: () => {
                      if (
                        presetNames.includes(newName) &&
                        !window.confirm(t('confirmOverwritePreset', newName))
                      )
                        return
                      send(
                        '/params/save',
                        { model, params: vals, preset: newName },
                        t('presetSavedActive', newName)
                      ).then(r => {
                        if (r && r.ok) setName('')
                      })
                    },
                    children: t('saveAsPreset')
                  })
                ]
              }),
              jsx('div', {
                className: 'text-[10px] text-(--ui-text-tertiary)',
                children:
                  t('appliedNextLoad') +
                  (model === current ? t('appliedNextLoadReload') : '.')
              }),
              note ? jsx('div', { className: 'text-[10px] text-(--ui-accent)', children: note }) : null
            ]
          }),
      jsx('div', {
        children: jsx(Button, { variant: 'ghost', size: 'sm', onClick: () => setOpen(false), children: t('close') })
      })
    ]
  })
}

// --------------------------------------------------------------------------- SmartModelProvider

// The model-provider half of this plugin. Only written to disk when the user
// asks for it here (or with `hermes sms provider install`).
// --------------------------------------------------------------------------- settings

const SECTION = 'text-[11px] font-medium uppercase tracking-wide text-(--ui-text-tertiary)'
const CARD = 'rounded-[4px] border border-(--ui-stroke-secondary)'
const HINT = 'text-[10px] leading-snug text-(--ui-text-tertiary)'

// One setting: what it is (and a one-line hint) on the left, its control on the right.
function SettingRow({ id, label, hint, error, children }) {
  return h(
    'div',
    { className: 'flex items-start gap-2 px-2 py-1.5', style: { borderTop: RULE } },
    h(
      'div',
      { className: 'flex min-w-0 flex-1 flex-col gap-0.5' },
      h('span', { id, className: 'text-xs' }, label),
      error ? h('span', { className: 'text-[10px] leading-snug text-destructive' }, error) : hint ? h('span', { className: HINT }, hint) : null
    ),
    h('div', { className: 'flex shrink-0 items-center gap-1', style: { paddingTop: 1 } }, children)
  )
}

function Toggle({ value, onChange, disabled, labelId }) {
  return jsx('button', {
    type: 'button',
    role: 'switch',
    'aria-checked': value ? 'true' : 'false',
    'aria-labelledby': labelId,
    title: value ? t('on') : t('off'),
    disabled,
    onClick: () => onChange(!value),
    className: cn(
      'relative w-8 shrink-0 rounded-full border transition-colors',
      value ? 'border-(--ui-accent) bg-(--ui-accent)' : 'border-(--ui-stroke-secondary) bg-transparent',
      disabled ? 'cursor-not-allowed opacity-50' : 'cursor-pointer'
    ),
    style: { height: 18 },
    children: jsx('span', {
      className: cn('absolute size-3 rounded-full transition-transform', value ? 'bg-background' : 'bg-(--ui-text-tertiary)'),
      style: { left: 2, top: 2, transform: value ? 'translateX(14px)' : 'none' }
    })
  })
}

function Segmented({ value, options, onChange, disabled, labelId }) {
  return jsx('div', {
    role: 'radiogroup',
    'aria-labelledby': labelId,
    className: 'inline-flex overflow-hidden rounded-[3px] border border-(--ui-stroke-secondary)',
    children: options.map(([id, label], i) =>
      jsx(
        'button',
        {
          type: 'button',
          role: 'radio',
          'aria-checked': id === value ? 'true' : 'false',
          disabled,
          onClick: () => (id === value ? null : onChange(id)),
          className: cn(
            'px-1.5 py-0.5 text-[11px] transition-colors',
            i ? 'border-l border-(--ui-stroke-secondary)' : '',
            id === value
              ? 'bg-(--ui-accent) text-background'
              : 'text-(--ui-text-tertiary) hover:bg-(--chrome-action-hover) hover:text-foreground',
            disabled ? 'cursor-not-allowed opacity-50' : 'cursor-pointer'
          ),
          children: label
        },
        id
      )
    )
  })
}

// What gpu_job.sh does with the LLM during a VRAM-heavy job: one option per
// line; the selected one opens to show what it does and what it means on this
// machine right now.
function GpuModePicker({ gpuModes, disabled, onPick }) {
  const list = gpuModes.modes || []
  const degraded = !!(gpuModes.layout && gpuModes.layout.degraded)
  return h(
    'div',
    { className: cn(CARD, 'flex flex-col') },
    jsx(Tip, {
      label: t('gpuModeTip'),
      children: h('span', { id: 'sms-gpu-mode', className: 'w-fit px-2 pt-1.5 text-[10px] font-medium text-(--ui-text-tertiary)' }, t('gpuModeTitle'))
    }),
    h(
      'div',
      { role: 'radiogroup', 'aria-labelledby': 'sms-gpu-mode', className: 'flex flex-col p-1' },
      list.map(m => {
        const on = m.id === gpuModes.mode
        return h(
          'button',
          {
            key: m.id,
            type: 'button',
            role: 'radio',
            'aria-checked': on ? 'true' : 'false',
            title: on ? undefined : m.description,
            disabled,
            onClick: () => (on ? null : onPick(m)),
            className: cn(
              'flex items-start gap-2 rounded-[3px] px-1.5 py-1 text-left transition-colors',
              on ? 'bg-(--chrome-action-hover)' : 'hover:bg-(--chrome-action-hover)',
              disabled ? 'cursor-not-allowed opacity-60' : on ? 'cursor-default' : 'cursor-pointer'
            )
          },
          h(
            'span',
            {
              className: cn(
                'flex size-3 shrink-0 items-center justify-center rounded-full border',
                on ? 'border-(--ui-accent)' : 'border-(--ui-text-tertiary)'
              ),
              style: { marginTop: 3 }
            },
            on ? h('span', { className: 'size-1.5 rounded-full bg-(--ui-accent)' }) : undefined
          ),
          h(
            'span',
            { className: 'flex min-w-0 flex-1 flex-col gap-0.5' },
            h('span', { className: cn('text-xs', on ? 'font-medium text-foreground' : 'text-(--ui-text-tertiary)') }, m.label),
            on ? h('span', { className: HINT }, m.description) : undefined,
            on && gpuModes.summary
              ? h(
                  'span',
                  { className: cn('text-[10px] leading-snug', degraded ? '' : 'text-foreground'), style: degraded ? WARN : undefined },
                  `${t('gpuModeNow')} ${gpuModes.summary}`
                )
              : undefined
          )
        )
      })
    )
  )
}

// The same behaviour settings the dashboard shows; each one saves on change.
function SettingsSection({ settings, gpuModes, disabled, onSave, onPickMode }) {
  const s = settings || {}
  const [idle, setIdle] = useState(null) // text being edited; null = show the saved value
  const idleText = idle == null ? String(s.idle_restore_min ?? '') : idle
  const idleNum = /^\d+$/.test(idleText.trim()) ? parseInt(idleText, 10) : NaN
  const idleBad = idle != null && !(idleNum >= 0 && idleNum <= 720)
  function commitIdle() {
    if (idle == null || idleBad) return
    if (idleNum === s.idle_restore_min) return setIdle(null)
    Promise.resolve(onSave('idle_restore_min', idleNum)).then(() => setIdle(null))
  }
  return h(
    'div',
    { className: 'flex flex-col gap-1.5' },
    h('div', { className: SECTION }, t('settingsHeader')),
    (gpuModes.modes || []).length ? h(GpuModePicker, { gpuModes, disabled, onPick: onPickMode }) : undefined,
    settings
      ? h(
          'div',
          { className: cn(CARD, 'flex flex-col') },
          h('span', { className: 'px-2 py-1.5 text-[10px] font-medium text-(--ui-text-tertiary)' }, t('automationTitle')),
          h(
            SettingRow,
            { id: 'sms-set-after', label: t('setAfterJob'), hint: t('setAfterJobHint') },
            h(Segmented, {
              labelId: 'sms-set-after',
              value: s.after_gpu_job,
              disabled,
              options: [['small', t('setAfterSmall')], ['primary', t('setAfterPrimary')]],
              onChange: v => onSave('after_gpu_job', v)
            })
          ),
          h(
            SettingRow,
            { id: 'sms-set-idle', label: t('setIdle'), hint: t('setIdleHint'), error: idleBad ? t('setIdleBad') : null },
            h('input', {
              'aria-labelledby': 'sms-set-idle',
              inputMode: 'numeric',
              value: idleText,
              disabled,
              onChange: e => setIdle(e.target.value),
              onBlur: commitIdle,
              onKeyDown: e => (e.key === 'Enter' ? commitIdle() : e.key === 'Escape' ? setIdle(null) : null),
              className: cn(
                'w-10 rounded-[3px] border bg-transparent px-1 py-0.5 text-right text-xs outline-none focus:border-(--ui-accent)',
                'border-(--ui-stroke-secondary)'
              ),
              style: idleBad ? BAD_BORDER : undefined
            }),
            h('span', { className: HINT }, t('setIdleUnit'))
          ),
          h(
            SettingRow,
            { id: 'sms-set-recover', label: t('setRecover'), hint: t('setRecoverHint') },
            h(Toggle, { labelId: 'sms-set-recover', value: !!s.auto_recover, disabled, onChange: v => onSave('auto_recover', v) })
          ),
          h(
            SettingRow,
            { id: 'sms-set-autostart', label: t('setAutostart'), hint: t('setAutostartHint') },
            h(Toggle, { labelId: 'sms-set-autostart', value: !!s.autostart_primary, disabled, onChange: v => onSave('autostart_primary', v) })
          )
        )
      : undefined
  )
}

function ProviderCard({ provider, onCall }) {
  const [pending, setPending] = useState(false)
  const [note, setNote] = useState(null)
  const pv = provider || {}
  const models = pv.models || []

  function send(path, okText) {
    setPending(true)
    setNote(null)
    return onCall(path, {})
      .then(r => {
        if (r && r.ok) setNote(okText)
      })
      .finally(() => setPending(false))
  }
  const muted = 'text-[10px] text-(--ui-text-tertiary)'

  return jsxs('div', {
    className: 'flex flex-col gap-1.5 rounded-[4px] border border-(--ui-stroke-secondary) p-2 text-xs',
    children: [
      jsxs('div', {
        className: 'flex items-center gap-1.5',
        children: [
          jsx(StatusDot, {
            tone: pv.installed ? (pv.in_sync ? 'good' : 'warn') : 'muted',
            className: 'size-2'
          }),
          jsx('span', { className: 'font-medium', children: 'SmartModelProvider' }),
          jsx('span', {
            className: cn('ml-auto', muted),
            children: pv.installed
              ? pv.in_sync
                ? t('providerInstalled')
                : t('providerOutdated')
              : pv.foreign
                ? t('providerFolderTaken')
                : t('providerNotInstalled')
          })
        ]
      }),
      pv.error ? jsx('div', { className: 'text-[10px] text-destructive', children: pv.error }) : null,
      pv.foreign
        ? jsx('div', {
            style: WARN, className: 'text-[10px]',
            children: t('providerForeign', pv.path)
          })
        : null,
      !pv.installed && !pv.foreign
        ? jsxs('div', {
            className: 'flex flex-col gap-1.5',
            children: [
              jsx('div', {
                className: muted,
                children:
                  t('providerExplain')
              }),
              jsx('div', {
                children: jsx(Button, {
                  size: 'sm',
                  disabled: pending,
                  onClick: () => send('/provider/install', t('providerInstalledMsg')),
                  children: t('installProvider')
                })
              }),
              jsx('div', {
                className: cn(muted, 'break-all'),
                children: t('providerCreatesFolder', pv.path || 'plugins/model-providers/smart-model-provider')
              })
            ]
          })
        : null,
      pv.installed
        ? jsxs('div', {
            className: 'flex flex-col gap-1.5',
            children: [
              jsx('div', {
                className: 'flex flex-wrap gap-1',
                children: models.map(m =>
                  jsx(
                    Badge,
                    {
                      variant: 'muted',
                      className: 'px-1 text-[10px]',
                      children: m.ctx ? `${m.name} · ${Math.round(m.ctx / 1024)}k` : m.name
                    },
                    m.name
                  )
                )
              }),
              jsx('div', {
                className: muted,
                children:
                  t('providerHowTo')
              }),
              jsxs('div', {
                className: 'flex items-center gap-1.5',
                children: [
                  !pv.in_sync
                    ? jsx(Button, {
                        size: 'sm',
                        disabled: pending,
                        onClick: () => send('/provider/install', t('providerUpdatedMsg')),
                        children: t('update')
                      })
                    : null,
                  jsx(Button, {
                    variant: 'ghost',
                    size: 'sm',
                    className: 'text-destructive',
                    disabled: pending,
                    onClick: () => {
                      if (window.confirm(t('confirmRemoveProvider')))
                        send('/provider/remove', t('providerRemovedMsg'))
                    },
                    children: t('remove')
                  })
                ]
              })
            ]
          })
        : null,
      note ? jsx('div', { className: 'text-[10px] text-(--ui-accent)', children: note }) : null
    ]
  })
}

// --------------------------------------------------------------------------- main pane

function SwapPane({ rest }) {
  const { locale } = useI18n()
  LANG = langOf(locale || navigator.language)
  const [editor, setEditor] = useState(null) // {name}: model open in the editor (null name = new)
  const [logOpen, setLogOpen] = useState(false)
  const [busy, setBusy] = useState(null) // 'swap' | 'restore'
  const [lastAction, setLastAction] = useState(null) // {kind, text}

  const query = useQuery({
    queryKey: ['sms', 'state'],
    queryFn: () => rest('/state'),
    refetchInterval: 5000,
    retry: 1
  })
  const data = query.data
  const status = (data && data.status) || {}
  const swap = status.swap || {}
  const cfg = (data && data.config) || {}
  const ui = (data && data.ui) || {}
  const engines = (data && data.engines) || []
  const slots = ui.slots || {}
  const models = (data && data.models) || []
  const health = (data && data.health) || {}
  const activeEngine = engines.find(e => e.active) || {}
  const current = (status.llm && status.llm.current) || '—'
  const stateName = swap.state || 'idle'
  const stateBusy = isBusy(stateName)
  const gpuJob = status.gpu_job || null
  const server = status.server || {}
  const comfy = status.comfyui || {}
  // only llama.cpp is managed today; Ollama / vLLM / LM Studio are under construction
  const managed = activeEngine.managed !== false
  const graceLeft =
    stateName === 'grace' && swap.grace_until
      ? Math.max(0, Math.ceil(swap.grace_until - Date.now() / 1000))
      : null
  const lastResult = swap.last_result || null
  const lastError =
    stateName === 'failed'
      ? swap.stale
        ? t('runnerDied')
        : (lastResult && lastResult.detail && lastResult.detail.error) || t('lastSwapFailed')
      : null

  // poll every second while a swap is in flight (grace countdown, live log)
  const refetch = query.refetch
  useEffect(() => {
    if (!stateBusy) return undefined
    const timer = setInterval(() => refetch(), 1000)
    return () => clearInterval(timer)
  }, [stateBusy, refetch])
  const comfyGpus = (comfy.vram || [])
    .filter(Boolean)
    .map(g => g.name || g.gpu || '')
    .filter(Boolean)
    .join(', ')

  function notify(kind, title, message) {
    try {
      host.notify({ kind, title, message })
    } catch (e) {
      /* host unavailable during hot reload */
    }
  }
  function flash(kind, text) {
    setLastAction({ kind, text, at: Date.now() })
  }
  async function call(path, body, busyKey, quiet) {
    if (busyKey) setBusy(busyKey)
    haptic('selection')
    try {
      const r = await rest(path, body ? { method: 'POST', body } : undefined)
      if (!r || r.ok === false) {
        if (!quiet) notify('error', 'Smart-Model-Swap', (r && r.error) || t('backendResponse'))
        haptic('error')
        return r
      }
      haptic('crisp')
      return r
    } catch (e) {
      haptic('error')
      notify('error', 'Smart-Model-Swap', String((e && e.message) || e))
      return null
    } finally {
      if (busyKey) setBusy(null)
      query.refetch()
    }
  }

  function dryRun() {
    call('/swap', { model: slots.secondary, dry_run: true }, 'dry', true).then(r => {
      if (!r) return
      const pf = r.preflight || {}
      const bad = (pf.checks || []).find(c => !c.ok)
      if (r.ok) flash('ok', t('dryRunOk', slots.secondary, pf.available_gb))
      else
        flash(
          'bad',
          t('dryRunFailed', bad ? `${bad.check} — ${bad.detail}` : r.error || (r.busy ? t('swapInProgress') : t('failed')))
        )
    })
  }

  const primary = cfg.primary_model || slots.primary || ''
  const secondary = slots.secondary || ''
  // Switch position: right = the secondary is (or is about to be) the loaded model.
  const onSecondary = stateBusy
    ? swap.action === 'swap'
    : current !== '—' && current !== primary
  const switchDisabled =
    stateBusy || !managed || !!busy || (!onSecondary && current !== '—' && (!secondary || secondary === primary))

  function toggle() {
    if (onSecondary || current === '—') {
      call('/restore', {}, 'restore').then(r => {
        if (r && r.ok) flash('ok', t('switchingToPrimary', primary))
      })
    } else {
      call('/swap', { model: secondary }, 'swap').then(r => {
        if (r && r.ok) flash('ok', t('switchingToSecondary', secondary))
      })
    }
  }
  function pickSlot(slot, name) {
    call('/slots', { slot, model: name }).then(r => {
      if (r && r.ok)
        flash(
          'ok',
          r.exchanged
            ? t('slotsExchanged')
            : `${slot === 'primary' ? t('primary') : t('secondary')}: ${name}`
        )
    })
  }
  // ---- settings (GPU job mode + automation): saved on change
  const gpuModes = (data && data.gpu_job_modes) || {}
  function pickGpuMode(m) {
    call('/gpu_job/mode', { mode: m.id }, 'gpumode').then(r => {
      if (r && r.ok) flash('ok', t('gpuModeSaved', m.label))
    })
  }
  function saveSetting(key, value) {
    return call('/settings', { key, value }, 'setting').then(r => {
      if (r && r.ok) flash('ok', t('setSaved'))
      return r
    })
  }
  const vram = Array.isArray(status.vram) ? status.vram : []

  function slotSelect(slot, value) {
    return jsxs('select', {
      value,
      disabled: stateBusy || !!busy,
      onChange: e => pickSlot(slot, e.target.value),
      className:
        'w-full min-w-0 rounded-[3px] border border-(--ui-stroke-secondary) bg-transparent px-1.5 py-1 text-xs outline-none focus:border-(--ui-accent)',
      children: [
        jsx('option', { value: '', disabled: true, children: t('choosePlaceholder') }, '__none'),
        models.map(m => jsx('option', { value: m.name, children: m.name }, m.name))
      ]
    })
  }
  const capClass = on =>
    cn(
      'text-[10px] font-medium uppercase tracking-wide',
      on ? 'text-(--ui-accent)' : 'text-(--ui-text-tertiary)'
    )

  return jsxs(
    'div',
    {
      className: 'flex h-full flex-col gap-3 overflow-y-auto p-3 text-sm',
      children: [
        // ---- header
        jsxs('div', {
          className: 'flex items-center gap-2',
          children: [
            jsx(StatusDot, { tone: stateTone(stateName) }),
            jsx('span', { className: 'text-xs font-semibold uppercase tracking-wide', children: 'Model Swap' }),
            jsx('span', {
              className: 'ml-auto text-[10px] text-(--ui-text-tertiary)',
              children: timeAgo(swap.updated_at)
            })
          ]
        }),

        // ---- current state
        jsxs(
          'div',
          {
            className: 'rounded-[6px] border border-(--ui-stroke-secondary) p-2.5',
            children: [
              jsxs('div', {
                className: 'flex items-center gap-2 text-sm font-medium',
                children: [
                  jsx(Codicon, { name: 'symbol-class', size: 14 }),
                  current === '—' && gpuJob ? t('gpuJobUnloaded') : current
                ]
              }),
              gpuJob
                ? jsx('div', {
                    style: WARN, className: 'mt-1 text-[11px]',
                    children: t(
                      current === '—' ? 'gpuJobUnloadedNote' : 'gpuJobRunningNote',
                      gpuJob.job,
                      Math.floor((gpuJob.elapsed_s || 0) / 60)
                    )
                  })
                : null,
              jsxs('div', {
                className: 'mt-1 flex flex-wrap items-center gap-x-3 gap-y-0.5 text-[11px] text-(--ui-text-tertiary)',
                children: [
                  jsx('span', {
                    children:
                      graceLeft != null
                        ? t('graceCutover', graceLeft, swap.target || '…')
                        : stateBusy
                        ? `${stateName} → ${swap.target || '…'}`
                        : t('stateLabel', stateName)
                  }),
                  jsx('span', {
                    children: t(
                      'freeVram',
                      status.total_free_gb != null ? freeGb(status.total_free_gb * 1024) : '?'
                    )
                  }),
                  activeEngine.label ? jsx('span', { children: t('engineLabel', activeEngine.label) }) : null,
                  server.summary ? jsx('span', { children: t('serverLabel', server.summary) }) : null,
                  comfy.up
                    ? jsx('span', { children: t('comfyOnGpu', comfyGpus ? ` (${comfyGpus})` : '') })
                    : null
                ]
              }),
              vram.length
                ? jsx('div', {
                    className: 'mt-1.5 flex flex-col gap-1',
                    children: vram.map(g =>
                      jsxs(
                        'div',
                        {
                          className: 'flex items-center gap-1.5 text-[10px] text-(--ui-text-tertiary)',
                          title: g.name,
                          children: [
                            jsx('span', { className: 'w-9 shrink-0', children: `GPU ${g.gpu}` }),
                            jsx('div', {
                              className: 'h-1 min-w-0 flex-1 overflow-hidden rounded-full bg-(--ui-stroke-secondary)',
                              children: jsx('div', {
                                className: 'h-full rounded-full bg-(--ui-accent)',
                                style: { width: `${g.total_mib ? Math.round((g.used_mib / g.total_mib) * 100) : 0}%` }
                              })
                            }),
                            jsx('span', {
                              className: 'shrink-0 tabular-nums',
                              children: t('vramOf', (g.used_mib / 1024).toFixed(1), freeGb(g.total_mib))
                            })
                          ]
                        },
                        g.gpu
                      )
                    )
                  })
                : null,
              lastError
                ? jsx('div', { className: 'mt-1 text-[11px] text-destructive', children: lastError })
                : null
            ]
          }
        ),

        // ---- primary <-> secondary switch, with a model picker on each side
        jsxs('div', {
          className: 'flex items-end gap-2',
          children: [
            jsxs('div', {
              className: 'flex min-w-0 flex-1 flex-col gap-0.5',
              children: [
                jsx('span', { className: capClass(!stateBusy && current === primary), children: t('primary') }),
                slotSelect('primary', primary)
              ]
            }),
            jsx('button', {
              type: 'button',
              role: 'switch',
              'aria-checked': onSecondary ? 'true' : 'false',
              'aria-label': t('switchAria'),
              title: onSecondary ? t('backToPrimary') : t('switchToSecondary'),
              disabled: switchDisabled,
              onClick: toggle,
              className: cn(
                'relative mb-0.5 h-6 shrink-0 rounded-full border border-(--ui-stroke-secondary) transition-colors',
                switchDisabled ? 'cursor-not-allowed opacity-50' : 'cursor-pointer hover:bg-(--chrome-action-hover)'
              ),
              style: { width: 44 },
              children: jsx('span', {
                className: 'absolute size-4 rounded-full bg-(--ui-accent) transition-transform',
                style: { left: 3, top: 3, transform: onSecondary ? 'translateX(20px)' : 'none' }
              })
            }),
            jsxs('div', {
              className: 'flex min-w-0 flex-1 flex-col items-end gap-0.5',
              children: [
                jsx('span', { className: capClass(!stateBusy && onSecondary), children: t('secondary') }),
                slotSelect('secondary', secondary)
              ]
            })
          ]
        }),

        jsxs('div', {
          className: 'flex items-center gap-2',
          children: [
            jsx(Button, {
              variant: 'outline',
              size: 'sm',
              loading: busy === 'dry',
              disabled: stateBusy || !secondary || !managed,
              onClick: dryRun,
              children: t('testSwap')
            }),
            busy === 'swap' || busy === 'restore' ? jsx(GlyphSpinner, { className: 'text-[11px]' }) : null
          ]
        }),

        !managed
          ? jsx('div', {
              className: 'text-[11px] text-(--ui-text-tertiary)',
              children: t('activeEngineWip')
            })
          : null,

        lastAction && Date.now() - lastAction.at < 12000
          ? jsx('div', {
              className: cn(
                'text-[11px]',
                lastAction.kind === 'bad' ? 'text-destructive' : 'text-(--ui-text-tertiary)'
              ),
              children: lastAction.text
            })
          : null,

        jsx(Separator, {}),

        // ---- settings: GPU job mode + automation
        h(SettingsSection, {
          settings: data && data.settings,
          gpuModes,
          disabled: !!busy,
          onSave: saveSetting,
          onPickMode: pickGpuMode
        }),

        jsx(Separator, {}),

        // ---- engines (SmartModelProvider)
        h(
          'div',
          { className: 'flex flex-col gap-1.5' },
          h('div', { className: 'text-[11px] font-medium uppercase tracking-wide text-(--ui-text-tertiary)' }, t('enginesHeader')),
          engines.map(e => h(EngineRow, { key: e.id, e, health: health[e.id], onCall: call, flash })),
          h('div', { className: 'text-[10px] text-(--ui-text-tertiary)' }, t('smpStartsEngine'))
        ),

        jsx(Separator, {}),

        // ---- registry + editor
        h(
          'div',
          { className: 'flex flex-col gap-1.5' },
          h('span', { className: 'text-[11px] font-medium uppercase tracking-wide text-(--ui-text-tertiary)' }, t('modelsHeader', models.length)),
          models.map(m =>
            h(ModelRow, {
              key: m.name,
              m,
              isPrimary: m.name === primary,
              isSecondary: m.name === slots.secondary,
              editing: !!editor && editor.name === m.name,
              onEdit: n => setEditor({ name: n })
            })
          ),
          editor
            ? h(ModelEditor, {
                key: editor.name || '__new__',
                name: editor.name,
                engines,
                primary,
                models,
                rest,
                onCall: call,
                flash,
                onSelect: n => setEditor({ name: n }),
                onClose: () => setEditor(null)
              })
            : h(
                'div',
                null,
                h(Button, { variant: 'outline', size: 'sm', onClick: () => setEditor({ name: null }) }, h('span', null, jsx(Codicon, { name: 'add', size: 12 }), ' ' + t('addModel')))
              )
        ),

        // ---- model parameters + presets
        jsx(ParamsEditor, {
          models,
          specs: (data && data.param_specs) || [],
          current,
          primary: cfg.primary_model || slots.primary || '',
          onCall: call
        }),

        // ---- SmartModelProvider (model-provider half)
        jsx(ProviderCard, { provider: data && data.provider, onCall: call }),

        // ---- log
        jsxs('div', {
          className: 'flex flex-col gap-1',
          children: [
            jsxs(
              'button',
              {
                type: 'button',
                onClick: () => setLogOpen(o => !o),
                className: 'flex items-center gap-1.5 text-[11px] font-medium uppercase tracking-wide text-(--ui-text-tertiary) hover:text-foreground',
                children: [
                  jsx(Codicon, { name: logOpen ? 'chevron-down' : 'chevron-right', size: 11 }),
                  t('runnerLog')
                ]
              }
            ),
            logOpen
              ? jsxs(
                  'pre',
                  {
                    className:
                      'max-h-40 overflow-y-auto rounded-[4px] border border-(--ui-stroke-secondary) p-2 font-mono text-[10px] leading-relaxed text-(--ui-text-tertiary)',
                    children: [
                      ((data && data.log_tail) || []).join('\n') || t('noLogsYet'),
                      stateBusy
                        ? jsx(GlyphSpinner, {
                            className: 'mt-1 inline-block text-[10px]'
                          })
                        : null
                    ]
                  }
                )
              : null
          ]
        })
      ]
    }
  )
}

// --------------------------------------------------------------------------- statusbar chip

function SwapChip({ rest }) {
  const { locale } = useI18n()
  LANG = langOf(locale || navigator.language)
  const query = useQuery({
    queryKey: ['sms', 'state'],
    queryFn: () => rest('/state'),
    refetchInterval: 5000,
    retry: 1
  })
  const data = query.data
  const status = (data && data.status) || {}
  const swap = status.swap || {}
  const stateName = swap.state || 'idle'
  const current = (status.llm && status.llm.current) || ''
  const label = isBusy(stateName)
    ? `${stateName} → ${swap.target || '…'}`
    : stateName === 'failed'
    ? t('swapFailed')
    : current || (status.gpu_job ? t('gpuJobShort') : 'idle')
  return jsx(
    Tip,
    {
      label: `Smart-Model-Swap — ${label}`,
      children: jsxs('button', {
        type: 'button',
        onClick: () => haptic('selection'),
        className: cn(
          'inline-flex h-full items-center gap-1.5 px-1.5 text-[0.6875rem] transition-colors',
          'text-(--ui-text-tertiary) hover:bg-(--chrome-action-hover) hover:text-foreground'
        ),
        children: [
          jsx(StatusDot, { tone: stateTone(stateName), className: 'size-2' }),
          jsx('span', { className: 'max-w-28 truncate', children: label })
        ]
      })
    }
  )
}

// --------------------------------------------------------------------------- registration

export default {
  id: ID,
  name: 'Smart-Model-Swap',
  register(ctx) {
    const rest =
      ctx.rest ||
      (() => Promise.reject(new Error(t('restUnavailable'))))
    const api = path => {
      // ctx.rest is already scoped to /api/plugins/<id>; pass through.
      return rest(path)
    }
    const apiPost = (path, body) => rest(path, { method: 'POST', body })
    const boundRest = (path, opts) => (opts ? apiPost(withLang(path), opts.body) : api(withLang(path)))

    ctx.register({
      id: 'pane',
      area: 'panes',
      title: 'Model Swap',
      data: { placement: 'right', width: '312px' },
      render: () => jsx(SwapPane, { rest: boundRest })
    })
    ctx.register({
      id: 'chip',
      area: 'statusBar.right',
      order: 120,
      render: () => jsx(SwapChip, { rest: boundRest })
    })
  }
}
