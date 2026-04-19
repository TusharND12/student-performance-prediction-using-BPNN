/** Labs API — proxied /api/labs/* in dev. */
import axios from 'axios';
import { apiUrl } from './apiRoutes.js';

function labsPath(p) {
  return apiUrl(`labs/${p}`);
}

export async function postSensitivity(payload) {
  const { data } = await axios.post(labsPath('sensitivity'), payload);
  return data;
}

export async function postCounterfactual(payload, field, value) {
  const { data } = await axios.post(labsPath('counterfactual'), { payload, field, value });
  return data;
}

export async function postAdversary(payload) {
  const { data } = await axios.post(labsPath('adversary'), payload);
  return data;
}

export async function getCohortSummary() {
  const { data } = await axios.get(labsPath('cohort/summary'));
  return data;
}

export async function postCohortCompare(payload) {
  const { data } = await axios.post(labsPath('cohort/compare'), payload);
  return data;
}

export async function getManifest() {
  const { data } = await axios.get(labsPath('manifest'));
  return data;
}

export async function postStudyBudget(body) {
  const { data } = await axios.post(labsPath('study-budget'), body);
  return data;
}

/** BPNN vs training-z shadow (Innovation Labs). */
export async function postShadowModel(payload) {
  const { data } = await axios.post(labsPath('shadow-model'), payload);
  return data;
}
