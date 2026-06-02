/**
 * page_help.js — centralised help descriptions for every GUI page.
 *
 * To update a description, edit the "text" field for the relevant page below.
 * The key matches the `active_page` value set by each Flask route.
 */

"use strict";

const AISCO_PAGE_HELP = {

  /* ── Introduction / pipeline-select ─────────────────────────────────── */
  "introduction": {
    title: "Home — Select Pipeline",
    text:
      "Choose which pipeline you want to use. " +
      "Live mode lets you configure and launch a new WP4 training run from scratch. " +
      "Archive (Completed) mode lets you browse and analyse finished runs without starting a new one.",
  },

  /* ── Launcher wizard ─────────────────────────────────────────────────── */
  "assembly-selection": {
    title: "Step 1 — Assembly Selection",
    text:
      "Pick the aeroshell geometry assembly you want to train on. " +
      "Each assembly defines the shape envelope and underlying mesh topology. " +
      "Confirm your choice and click Next to proceed to parameter configuration.",
  },

  "config-overrides": {
    title: "Step 2 — Configure & Run",
    text:
      "Review and adjust the key training hyperparameters before launching. " +
      "Any value left unchanged will use the project default. " +
      "Click Run to start the WP4 model training on the selected assembly. " +
      "The training progress indicator in the sidebar will activate once the run has started.",
  },

  "output": {
    title: "Step 3 — Output",
    text:
      "Monitor the active training run in real time. " +
      "Mesh visualisations and performance metrics update automatically as each epoch completes. " +
      "Use the epoch scrubber at the bottom to step back through earlier epochs and compare shapes. " +
      "When the run finishes you will be forwarded automatically to the leaderboard.",
  },

  "leaderboard": {
    title: "Step 4 — Live Leaderboard",
    text:
      "Global ranking of all completed runs, sorted by cruising speed (highest first). " +
      "The current run appears as soon as it produces its first result. " +
      "Use this view to compare the current run's performance against all historical results at a glance.",
  },

  /* ── Archive / viewer ────────────────────────────────────────────────── */
  "runs": {
    title: "Archive — Finished Runs",
    text:
      "Browse all completed training runs stored on the server. " +
      "Click any row to open it in the viewer and explore its meshes, metrics and visualisations. " +
      "Runs are sorted by most recent first.",
  },

  "latent-explorer": {
    title: "Latent Space Explorer",
    text:
      "Explore the latent space of a completed GINN model. " +
      "Move the scrubber to traverse the learned shape manifold and observe how the geometry " +
      "and aerodynamic performance metrics change across the latent dimension. " +
      "Use this to understand which parts of latent space produce well-performing shapes.",
  },

};
