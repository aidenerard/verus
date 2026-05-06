export const SERVER = import.meta.env.VITE_API_URL !== undefined
  ? import.meta.env.VITE_API_URL
  : 'https://verus-server.onrender.com';

export const MAPBOX_TOKEN = import.meta.env.VITE_MAPBOX_TOKEN ?? '';
export const DEFAULT_CENTER: [number, number] = [-73.9519, 40.8517];

export const GPR_EXTS = new Set([
  '.csv', '.dzt', '.dt1', '.rd3', '.rd7', '.segy', '.sgy', '.dzg', '.hd', '.rad',
  '.dt', '.gec', '.iprb', '.iprh',
]);

// ── Design tokens ─────────────────────────────────────────────────────────────

export const BG      = '#F5F3EF';
export const PANEL   = '#FFFFFF';
export const RAISED  = '#F5F3EF';
export const BORDER  = '#E2DED9';
export const BORDER2 = '#C8C3BD';
export const TEXT    = '#0A0A0A';
export const TEXT2   = '#7A7470';
export const ACCENT  = '#E8601C';

// ── Timing ────────────────────────────────────────────────────────────────────

export const STATUS_MSGS = [
  'Waking up server…', 'Loading AI model…',
  'Running inference…', 'Generating C-scan…', 'Almost done…',
];
export const WAKE_TIMEOUT_MS  = 3 * 60 * 1000;
export const WAKE_INTERVAL_MS = 4_000;
export const POLL_TIMEOUT_MS  = 8 * 60 * 1000;

// ── Manufacturer / frequency data ─────────────────────────────────────────────

export const MANUFACTURERS = [
  { key: 'gssi',             name: 'GSSI',               formats: '.dzt, .dzx',    series: 'BridgeScan · SIR series' },
  { key: 'sensors_software', name: 'Sensors & Software', formats: '.dt1, .hd',     series: 'Pulse EKKO · Noggin · LMX' },
  { key: 'mala',             name: 'MALA Geoscience',    formats: '.rd3, .rd7',    series: 'Easy Locator · ProEx · CX' },
  { key: 'ids',              name: 'IDS GeoRadar',       formats: '.dt, .gec',     series: 'RIS series' },
  { key: 'impulseradar',     name: 'ImpulseRadar',       formats: '.iprb, .iprh',  series: 'Raptor · Cobra' },
  { key: 'segy',             name: 'Other / SEG-Y',      formats: '.sgy, .segy',   series: 'Universal format' },
  { key: 'csv',              name: 'Processed CSV',      formats: '.csv',          series: 'Pre-processed export' },
] as const;

export type ManufacturerKey = typeof MANUFACTURERS[number]['key'];

export const FREQ_OPTIONS = [
  { mhz: 400,  label: '400 MHz',  desc: 'Deep penetration — pavement/utility (>1m)' },
  { mhz: 900,  label: '900 MHz',  desc: 'Medium depth — general bridge deck' },
  { mhz: 1600, label: '1600 MHz', desc: 'High resolution — shallow concrete (most common)' },
  { mhz: 2000, label: '2000 MHz', desc: 'Ultra high resolution — thin concrete cover' },
  { mhz: 2600, label: '2600 MHz', desc: 'Maximum resolution — surface features' },
] as const;

export const MANUFACTURER_EXTS: Record<string, string> = {
  gssi:             '.dzt,.DZT',
  sensors_software: '.dt1,.DT1,.hd',
  mala:             '.rd3,.rd7,.rad',
  ids:              '.dt,.gec',
  impulseradar:     '.iprb,.iprh',
  segy:             '.sgy,.segy',
  csv:              '.csv',
};

export const DEFAULT_ER: Record<number, number> = { 400: 8, 900: 7, 1600: 6, 2000: 6, 2600: 5 };

// ── Layer definitions ─────────────────────────────────────────────────────────

import { Radio, Layers } from 'lucide-react';

export const LAYER_DEFS = [
  { id: 'gpr',         label: 'GPR Profiles',     Icon: Radio },
  { id: 'condition',   label: 'Condition Grid',    Icon: Layers },
  { id: 'amplitude',   label: 'Amplitude Grid',    Icon: Layers },
  { id: 'satellite',   label: 'Satellite Image',   Icon: Layers },
  { id: 'annotations', label: 'Point Annotations', Icon: Layers },
] as const;

export type LayerId = typeof LAYER_DEFS[number]['id'];
