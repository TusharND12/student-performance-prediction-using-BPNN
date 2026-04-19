import axios from 'axios';
import { apiUrl } from './apiRoutes.js';

function metaPath(p) {
  return apiUrl(p);
}

export async function getMetaVersion() {
  const { data } = await axios.get(metaPath('meta/version'));
  return data;
}

export async function postCompleteness(payload) {
  const { data } = await axios.post(metaPath('meta/completeness'), payload);
  return data;
}

export async function getCalibration() {
  const { data } = await axios.get(metaPath('meta/calibration'));
  return data;
}

export async function getEnsembleStatus() {
  const { data } = await axios.get(metaPath('meta/ensemble'));
  return data;
}

export async function postGoalSeek(body) {
  const { data } = await axios.post(metaPath('labs/goal-seek'), body);
  return data;
}

export async function postBatchPredict(rows) {
  const { data } = await axios.post(metaPath('predict/batch'), { rows });
  return data;
}
