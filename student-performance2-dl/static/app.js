(() => {
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const ic = n => `<i data-lucide="${n}"></i>`;
const icons = () => { if (window.lucide) lucide.createIcons(); };

/* ---------- Default OULAD Schema Fallback ---------- */
const DEFAULT_SECTIONS = [
  {
    title: 'Student Profile', icon: 'user-round', sub: 'Demographic & background information from university registry',
    fields: [
      {k:'gender', label:'Gender', type:'seg', opts:[['F','Female'],['M','Male']], def:'M'},
      {k:'age_band', label:'Age Band', type:'seg', opts:[['0-35','Under 35'],['35-55','35 to 55'],['55<=','55 or older']], def:'0-35'},
      {k:'highest_education', label:'Highest Prior Qualification', type:'seg', opts:[['No Formal quals','No formal'],['Lower Than A Level','Below A-level'],['A Level or Equivalent','A-level'],['HE Qualification','Degree'],['Post Graduate Qualification','Postgrad']], def:'A Level or Equivalent', full:true},
      {k:'region', label:'Geographic Region', type:'select', opts:[['East Anglian Region','East Anglia'],['East Midlands Region','East Midlands'],['Ireland','Ireland'],['London Region','London'],['North Region','North'],['North Western Region','North West'],['Scotland','Scotland'],['South East Region','South East'],['South Region','South'],['South Western Region','South West'],['Wales','Wales'],['West Midlands Region','West Midlands'],['Yorkshire Region','Yorkshire']], def:'East Anglian Region'},
      {k:'imd_band', label:'Area Deprivation (IMD Band)', type:'select', opts:[['Unknown','Unknown'],['0-10%','0-10% (Most deprived)'],['10-20%','10-20%'],['20-30%','20-30%'],['30-40%','30-40%'],['40-50%','40-50%'],['50-60%','50-60%'],['60-70%','60-70%'],['70-80%','70-80%'],['80-90%','80-90%'],['90-100%','90-100% (Least deprived)']], def:'20-30%'},
      {k:'disability', label:'Declared Disability', type:'toggle', def:'N', vals:['Y','N']}
    ]
  },
  {
    title: 'Course & Enrollment', icon: 'book-open', sub: 'Curriculum module, study load, and registration timing',
    fields: [
      {k:'code_module', label:'Module Code', type:'seg', opts:[['AAA','AAA'],['BBB','BBB'],['CCC','CCC'],['DDD','DDD'],['EEE','EEE'],['FFF','FFF'],['GGG','GGG']], def:'BBB', full:true},
      {k:'semester', label:'Intake Semester', type:'seg', opts:[['B','February (B)'],['J','October (J)']], def:'J'},
      {k:'num_of_prev_attempts', label:'Previous Module Attempts', type:'slider', min:0, max:6, step:1, def:0, lo:'First attempt', hi:'6 or more'},
      {k:'studied_credits', label:'Total Credits Studied Concurrently', type:'slider', min:30, max:360, step:30, def:60, lo:'30 credits', hi:'360 credits'},
      {k:'reg_days_before', label:'Days Registered Before Course Start', type:'slider', min:-30, max:250, step:1, def:55, lo:'After start', hi:'250 days early'}
    ]
  },
  {
    title: 'Online Learning Engagement (First 60 Days)', icon: 'mouse-pointer-click', sub: 'Virtual Learning Environment (VLE) interaction patterns',
    fields: [
      {k:'active_days', label:'Days Active on Platform', type:'slider', min:0, max:60, step:1, def:25, lo:'0 days', hi:'60 days'},
      {k:'n_sites', label:'Distinct Learning Resources Visited', type:'slider', min:0, max:120, step:1, def:35, lo:'0 materials', hi:'120+'},
      {k:'clicks_total', label:'Total Platform Clicks', type:'slider', min:0, max:3500, step:25, def:520, lo:'0 clicks', hi:'3500+'},
      {k:'clicks_content', label:'Course Content Clicks', type:'slider', min:0, max:1800, step:20, def:260, lo:'0', hi:'1800+'},
      {k:'clicks_quiz', label:'Formative Quiz Clicks', type:'slider', min:0, max:400, step:10, def:45, lo:'0', hi:'400+'},
      {k:'clicks_forum', label:'Discussion Forum Clicks', type:'slider', min:0, max:800, step:10, def:85, lo:'0', hi:'800+'},
      {k:'clicks_home', label:'Homepage Visits', type:'slider', min:0, max:600, step:10, def:110, lo:'0', hi:'600+'},
      {k:'clicks_resource', label:'Static Resource Clicks', type:'slider', min:0, max:250, step:5, def:20, lo:'0', hi:'250+'}
    ]
  },
  {
    title: 'Early Assessment Performance (First 60 Days)', icon: 'file-check', sub: 'Continuous coursework results on TMA/CMA assessments before day 60',
    fields: [
      {k:'n_assessments_submitted', label:'Assessments Submitted', type:'slider', min:0, max:5, step:1, def:2, lo:'0', hi:'5'},
      {k:'avg_assessment_score', label:'Average Early Assessment Score', type:'slider', min:0, max:100, step:1, def:72, lo:'0 (Fail)', hi:'100 (Distinction)'},
      {k:'assessment_score_std', label:'Score Variability (Std Dev)', type:'slider', min:0, max:30, step:1, def:5, lo:'Consistent', hi:'Variable'},
      {k:'avg_days_early', label:'Avg Days Submitted Ahead of Due Date', type:'slider', min:-20, max:25, step:1, def:3, lo:'Late', hi:'25 days early'},
      {k:'late_submissions', label:'Number of Overdue Submissions', type:'slider', min:0, max:4, step:1, def:0, lo:'0', hi:'4+'},
      {k:'assessment_completion_rate', label:'Assessment Completion Ratio', type:'slider', min:0, max:1, step:0.1, def:1.0, lo:'0%', hi:'100%'}
    ]
  }
];

let META = {
  scale: 100,
  pass_mark: 40,
  good_mark: 70,
  dataset: 'oulad',
  verified_student_count: 23985,
  verified_record_count: 26361,
  selected_model: 'Hybrid Ensemble (DL + Boost)',
  test_rmse: 18.43,
  test_mae: 13.76,
  test_r2: 0.6333,
  sections: DEFAULT_SECTIONS
};
let RES = null;
const state = {};
const LABEL = {};

const allSections = () => META.sections || DEFAULT_SECTIONS;
const buildLabels = () => allSections().forEach(s => s.fields.forEach(f => LABEL[f.k] = f.label));

/* ---------- Field Renderers ---------- */
function renderField(f) {
  state[f.k] = f.def;
  if (f.type === 'slider') {
    const pct = ((f.def - f.min) / (f.max - f.min)) * 100;
    return `
      <div class="field">
        <div class="lbl">
          <label>${f.label}</label>
          <b class="val" id="v-${f.k}">${f.def}</b>
        </div>
        <input class="range" type="range" data-k="${f.k}" min="${f.min}" max="${f.max}" step="${f.step || 1}" value="${f.def}" style="--pct:${pct}%">
        <div class="hint"><span>${f.lo ?? f.min}</span><span>${f.hi ?? f.max}</span></div>
      </div>`;
  }
  if (f.type === 'seg') {
    return `
      <div class="field ${f.full ? 'full' : ''}">
        <div class="lbl"><label>${f.label}</label></div>
        <div class="seg" data-k="${f.k}">
          ${f.opts.map(([v, t]) =>
            `<button type="button" data-v="${v}" class="${String(v) === String(f.def) ? 'on' : ''}">${t}</button>`
          ).join('')}
        </div>
      </div>`;
  }
  if (f.type === 'select') {
    return `
      <div class="field ${f.full ? 'full' : ''}">
        <div class="lbl"><label>${f.label}</label></div>
        <div class="sel">
          <select data-k="${f.k}">
            ${f.opts.map(([v, t]) =>
              `<option value="${v}" ${String(v) === String(f.def) ? 'selected' : ''}>${t}</option>`
            ).join('')}
          </select>
        </div>
      </div>`;
  }
  const [on, off] = f.vals || ['Y', 'N'];
  return `
    <div class="field tog">
      <label>${f.label}</label>
      <label class="switch">
        <input type="checkbox" data-k="${f.k}" data-on="${on}" data-off="${off}" ${f.def === on ? 'checked' : ''}>
        <i></i>
      </label>
    </div>`;
}

function buildForm() {
  $('#form').innerHTML = allSections().map(s => `
    <section class="glass card">
      <h2><span class="ibox">${ic(s.icon)}</span>${s.title}</h2>
      <p class="sub">${s.sub}</p>
      <div class="grid">${s.fields.map(renderField).join('')}</div>
    </section>
  `).join('');
  icons();
}

/* ---------- Event Listeners ---------- */
$('#form').addEventListener('input', e => {
  const t = e.target;
  if (!t.matches('.range')) return;
  state[t.dataset.k] = +t.value;
  t.style.setProperty('--pct', ((t.value - t.min) / (t.max - t.min)) * 100 + '%');
  const vEl = $('#v-' + t.dataset.k);
  if (vEl) vEl.textContent = t.value;
  schedulePrediction();
});

$('#form').addEventListener('change', e => {
  const t = e.target;
  if (t.matches('.switch input')) {
    state[t.dataset.k] = t.checked ? t.dataset.on : t.dataset.off;
    schedulePrediction();
  } else if (t.matches('select')) {
    state[t.dataset.k] = t.value;
    schedulePrediction();
  }
});

$('#form').addEventListener('click', e => {
  const b = e.target.closest('.seg button');
  if (!b) return;
  const seg = b.parentElement;
  $$('button', seg).forEach(x => x.classList.remove('on'));
  b.classList.add('on');
  state[seg.dataset.k] = b.dataset.v;
  schedulePrediction();
});

$('#reset').addEventListener('click', () => {
  buildForm();
  executePrediction();
});

/* ---------- Prediction Engine ---------- */
const getStatusObj = score => {
  if (score < META.pass_mark) {
    return { name: 'At Risk', color: '#f87171', icon: 'triangle-alert' };
  } else if (score < META.good_mark) {
    return { name: 'Average', color: '#fbbf24', icon: 'circle-minus' };
  } else {
    return { name: 'Good / Distinction', color: '#34d399', icon: 'circle-check' };
  }
};

let timerId = null;
let requestSeq = 0;

function schedulePrediction() {
  clearTimeout(timerId);
  timerId = setTimeout(executePrediction, 100);
}

async function executePrediction() {
  const currentSeq = ++requestSeq;
  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(state)
    });
    if (!res.ok) throw new Error('API request failed');
    const data = await res.json();
    if (currentSeq !== requestSeq) return;
    displayPredictionResult(data);
  } catch (err) {
    $('#gstatus').textContent = 'Server recalculating…';
  }
}

const OPERATORS = {
  '<': (a, b) => a < b,
  '<=': (a, b) => a <= b,
  '>': (a, b) => a > b,
  '>=': (a, b) => a >= b,
  '==': (a, b) => a == b
};

function generateEducationalInsights(s) {
  const insights = [];
  const add = (t, k) => insights.push({ t, k });

  if (META.rules && Array.isArray(META.rules)) {
    META.rules.forEach(r => {
      if (s[r.f] !== undefined && OPERATORS[r.op] && OPERATORS[r.op](s[r.f], r.v)) {
        add(r.t, r.k);
      }
    });
  }

  if (!insights.length) {
    add('Student indicators align with regular cohort progress. No critical warning flags.', 'ok');
  }
  return insights;
}

function displayPredictionResult(resData) {
  const score = resData.grade;
  const statusObj = getStatusObj(score);
  const unc = resData.uncertainty || {};
  const rmse = unc.test_rmse || META.test_rmse || 18.4;
  const interval = unc.interval_68 || [Math.max(0, score - rmse).toFixed(1), Math.min(META.scale, score + rmse).toFixed(1)];

  const resultContainer = $('#result');
  resultContainer.style.setProperty('--c', statusObj.color);

  $('#gnum').textContent = score.toFixed(1);
  const ringCircumference = 364.4;
  $('#ringfg').style.strokeDasharray = `${(score / META.scale) * ringCircumference} ${ringCircumference}`;
  $('#gstatus').innerHTML = `${ic(statusObj.icon)}${statusObj.name}`;
  $('#gest').textContent = `Expected Score: ${score.toFixed(1)} / ${META.scale}`;

  // Statistically Grounded Prediction Interval
  $('#gci').innerHTML = `
    <b>Statistical Confidence (68% Interval)</b>
    <span>${interval[0]} – ${interval[1]} points (±${rmse.toFixed(1)} RMSE on unseen test students)</span>
  `;

  // Scale marker
  $('#gmark').style.left = `${(score / META.scale) * 100}%`;

  // Insights
  const iconMap = { bad: 'triangle-alert', warn: 'lightbulb', ok: 'circle-check' };
  $('#ins').innerHTML = generateEducationalInsights(state).map(x => `
    <li class="${x.k}">${ic(iconMap[x.k])}<span>${x.t}</span></li>
  `).join('');
  icons();
}

/* ---------- Batch Cohort Prediction ---------- */
const dropZone = $('#drop');
const fileInput = $('#file');

dropZone.addEventListener('click', () => fileInput.click());
['dragenter', 'dragover'].forEach(ev => {
  dropZone.addEventListener(ev, e => { e.preventDefault(); dropZone.classList.add('over'); });
});
['dragleave', 'drop'].forEach(ev => {
  dropZone.addEventListener(ev, e => { e.preventDefault(); dropZone.classList.remove('over'); });
});
dropZone.addEventListener('drop', e => {
  if (e.dataTransfer.files[0]) uploadCohortFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) uploadCohortFile(fileInput.files[0]);
});
$('#sample').addEventListener('click', () => uploadCohortFile(null));

async function uploadCohortFile(file) {
  $('#bmsg').textContent = 'Processing student records through pipeline…';
  let response;
  if (file) {
    const formData = new FormData();
    formData.append('file', file);
    response = await fetch('/api/batch', { method: 'POST', body: formData });
  } else {
    response = await fetch('/api/batch?sample=1', { method: 'POST' });
  }

  const result = await response.json();
  if (!response.ok) {
    $('#bmsg').textContent = result.error || 'Cohort processing failed';
    return;
  }

  $('#bmsg').textContent = `✓ ${result.rows.length.toLocaleString()} students evaluated`;
  renderCohortTable(result);
}

const statusChip = s => {
  const col = s === 'At risk' ? '#f87171' : (s === 'Average' ? '#fbbf24' : '#34d399');
  return `<span class="chip" style="--c:${col}">${s}</span>`;
};

function renderCohortTable(batchData) {
  const rows = batchData.rows;
  const summary = batchData.summary || {
    total_students: rows.length,
    mean_predicted_score: (rows.reduce((acc, r) => acc + (r.Predicted_Score || r.Predicted_G3 || 0), 0) / rows.length).toFixed(1),
    at_risk_count: rows.filter(r => r.Status === 'At risk').length,
    average_count: rows.filter(r => r.Status === 'Average').length,
    good_count: rows.filter(r => r.Status === 'Good' || r.Status === 'Good / Distinction').length,
  };

  const displayCols = (META.batch_cols || [
    'code_module', 'gender', 'age_band', 'avg_assessment_score',
    'active_days', 'clicks_total', 'Predicted_Score', 'Interval_68', 'Status'
  ]).filter(c => c in rows[0] || c === 'Interval_68');

  $('#bout').innerHTML = `
    <div class="stats">
      <div class="glass stat"><span>Cohort Size</span><b>${summary.total_students.toLocaleString()}</b></div>
      <div class="glass stat"><span>Mean Score</span><b>${summary.mean_predicted_score}</b></div>
      <div class="glass stat"><span>At Risk (&lt;40)</span><b style="color:#f87171">${summary.at_risk_count.toLocaleString()}</b></div>
      <div class="glass stat"><span>Good / Distinction</span><b style="color:#34d399">${summary.good_count.toLocaleString()}</b></div>
    </div>
    <div class="glass card">
      <div class="tablewrap">
        <table>
          <thead>
            <tr>
              ${displayCols.map(c => `<th>${c === 'Predicted_Score' || c === 'Predicted_G3' ? 'Predicted Score' : (c === 'Interval_68' ? '68% Conf. Interval' : (LABEL[c] || c.replace(/_/g, ' ')))}</th>`).join('')}
            </tr>
          </thead>
          <tbody>
            ${rows.slice(0, 300).map(r => `
              <tr>
                ${displayCols.map(c => {
                  if (c === 'Status') return `<td>${statusChip(r[c])}</td>`;
                  if (c === 'Predicted_Score' || c === 'Predicted_G3') return `<td><b>${(r[c] !== undefined ? r[c] : r.Predicted_G3).toFixed(1)}</b></td>`;
                  return `<td>${r[c] !== undefined ? r[c] : '–'}</td>`;
                }).join('')}
              </tr>
            `).join('')}
          </tbody>
        </table>
      </div>
      <div class="row">
        <button class="btn" id="dl">${ic('download')}Export Full Cohort Predictions (CSV)</button>
        <span class="muted">${rows.length > 300 ? `Showing preview of 300 / ${rows.length.toLocaleString()} rows.` : ''}</span>
      </div>
    </div>
  `;
  icons();

  $('#dl').onclick = () => {
    const keys = Object.keys(rows[0]);
    const escapeCsv = v => `"${String(v ?? '').replace(/"/g, '""')}"`;
    const csvContent = [keys.join(','), ...rows.map(r => keys.map(k => escapeCsv(r[k])).join(','))].join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = 'oulad_student_predictions.csv';
    link.click();
  };
}

/* ---------- Performance Dashboard ---------- */
async function loadPerformanceSection() {
  if (!RES) {
    try {
      const res = await fetch('/api/results');
      if (res.ok) RES = await res.json();
    } catch (e) {}
  }
  if (!RES || !RES.results) {
    $('#perf').innerHTML = '<p class="muted">Loading benchmark results…</p>';
    return;
  }

  const results = RES.results;
  const modelNames = Object.keys(results);
  const selectedModel = RES.selected_model || META.selected_model || modelNames[0];
  const verifiedStudents = RES.verified_student_count || META.verified_student_count || 23985;
  const verifiedRecords = RES.verified_record_count || META.verified_record_count || 26361;

  const topImportance = Object.entries(RES.importance || {}).slice(0, 10);
  const maxImp = Math.max(...topImportance.map(x => x[1]), 1e-5);

  const imageDescriptions = {
    'pred_vs_actual.png': 'Predicted vs Actual Scatter Plot',
    'loss_curve.png': 'Neural Network Training & Validation Loss',
    'residual_dist.png': 'Holdout Residual Error Distribution',
    'error_analysis.png': 'MAE Error Across Performance Tiers',
    'model_comparison.png': 'Comparative R² and MAE Benchmark',
    'feature_importance.png': 'Permutation Feature Importance',
    'eda_g3_dist.png': 'OULAD Final Score Distribution',
    'eda_corr.png': 'Key Feature Correlation Heatmap'
  };

  $('#perf').innerHTML = `
    <!-- Top Summary Stat Cards -->
    <div class="summary-grid">
      <div class="glass card summary-card">
        <span>Selected Model</span>
        <b style="font-size:1.45rem">${selectedModel}</b>
        <small>Primary production regressor</small>
      </div>
      <div class="glass card summary-card">
        <span>Test R² Score</span>
        <b style="color:#6c74ff">${results[selectedModel]?.R2?.toFixed(4) || '–'}</b>
        <small>Holdout unseen students</small>
      </div>
      <div class="glass card summary-card">
        <span>Test MAE Error</span>
        <b style="color:#34d399">${results[selectedModel]?.MAE?.toFixed(2) || '–'} pts</b>
        <small>Scale 0–100</small>
      </div>
      <div class="glass card summary-card">
        <span>Test RMSE Error</span>
        <b style="color:#fbbf24">${results[selectedModel]?.RMSE?.toFixed(2) || '–'} pts</b>
        <small>Holdout standard error</small>
      </div>
      <div class="glass card summary-card">
        <span>Verified Students</span>
        <b>${verifiedStudents.toLocaleString()}</b>
        <small>${verifiedRecords.toLocaleString()} course enrollments</small>
      </div>
    </div>

    <!-- Comparative Table -->
    <div class="glass card" style="margin-bottom:24px">
      <h2><span class="ibox">${ic('table')}</span>Model Comparison Benchmark (Holdout Test Set)</h2>
      <p class="sub">Evaluated strictly on completely unseen student records (zero student overlap with training data)</p>
      <div class="perf-table-wrap">
        <table class="perf-table">
          <thead>
            <tr>
              <th>Model Architecture</th>
              <th>R² Score</th>
              <th>MAE</th>
              <th>RMSE</th>
              <th>Within ±10 pts</th>
              <th>Within ±15 pts</th>
              <th>Role</th>
            </tr>
          </thead>
          <tbody>
            ${modelNames.map(name => {
              const m = results[name];
              const isHl = name === selectedModel;
              return `
                <tr class="${isHl ? 'highlight' : ''}">
                  <td>${name} ${isHl ? '★' : ''}</td>
                  <td><b>${m.R2?.toFixed(4)}</b></td>
                  <td>${m.MAE?.toFixed(2)}</td>
                  <td>${m.RMSE?.toFixed(2)}</td>
                  <td>${m['Within_10pts_%'] !== undefined ? m['Within_10pts_%'] + '%' : '–'}</td>
                  <td>${m['Within_15pts_%'] !== undefined ? m['Within_15pts_%'] + '%' : '–'}</td>
                  <td>${name.includes('Ensemble') ? 'Production' : (name.includes('Deep') ? 'Deep Learning' : 'Baseline')}</td>
                </tr>
              `;
            }).join('')}
          </tbody>
        </table>
      </div>
    </div>

    <!-- Bar Charts (R² & Feature Importance) -->
    <div class="two">
      <div class="glass card">
        <h2><span class="ibox">${ic('bar-chart-3')}</span>R² Performance by Model</h2>
        <p class="sub">Test set R² regression score (higher is better)</p>
        <div class="bars">
          ${modelNames.map(name => `
            <div class="bar">
              <div class="t"><span>${name}</span><span>${results[name]?.R2?.toFixed(4)}</span></div>
              <div class="trk"><div class="fill" style="width:${Math.max(0, results[name]?.R2 || 0) * 100}%"></div></div>
            </div>
          `).join('')}
        </div>
      </div>
      <div class="glass card">
        <h2><span class="ibox">${ic('trending-up')}</span>Top Student Performance Predictors</h2>
        <p class="sub">Increase in MAE when feature is permuted / shuffled</p>
        <div class="bars">
          ${topImportance.map(([col, val]) => `
            <div class="bar">
              <div class="t"><span>${LABEL[col] || col.replace(/_/g, ' ')}</span><span>+${val.toFixed(2)}</span></div>
              <div class="trk"><div class="fill" style="width:${(Math.max(0, val) / maxImp) * 100}%"></div></div>
            </div>
          `).join('')}
        </div>
      </div>
    </div>

    <!-- Generated Visualizations -->
    <div class="two">
      ${(RES.images || []).filter(img => imageDescriptions[img]).map(img => `
        <div class="glass card imgcard">
          <h2>${imageDescriptions[img]}</h2>
          <img src="/outputs/${img}" alt="${imageDescriptions[img]}">
        </div>
      `).join('')}
    </div>
  `;
  icons();
}

/* ---------- Navigation ---------- */
function handleRouting() {
  let view = location.hash.slice(1);
  if (!['predict', 'batch', 'performance', 'about'].includes(view)) view = 'predict';
  $$('.view').forEach(x => x.classList.toggle('active', x.id === 'view-' + view));
  $$('nav a').forEach(a => a.classList.toggle('active', a.dataset.view === view));
  if (view === 'performance') loadPerformanceSection();
  window.scrollTo({ top: 0 });
}
window.addEventListener('hashchange', handleRouting);

/* ---------- Dynamic Metadata Binding ---------- */
function bindMetadata() {
  const scale = META.scale || 100;
  const pass = (META.pass_mark / scale) * 100;
  const good = (META.good_mark / scale) * 100;

  const b1 = $('.band .s1'), b2 = $('.band .s2'), b3 = $('.band .s3');
  if (b1) b1.style.width = pass + '%';
  if (b2) b2.style.width = (good - pass) + '%';
  if (b3) b3.style.width = (100 - good) + '%';

  const scaleLabel = $('#gscale');
  if (scaleLabel) scaleLabel.textContent = `out of ${scale}`;
  if (META.title) $('#ph1').textContent = META.title;
  if (META.lead) $('#plead').textContent = META.lead;
  if (META.target_label) $('#rlabel').textContent = META.target_label;
  if (META.about_data) $('#about-data').textContent = META.about_data;
  if (META.about_model) $('#about-model').textContent = META.about_model;
}

/* ---------- App Initialization ---------- */
(async () => {
  try {
    const metaRes = await fetch('/api/meta');
    if (metaRes.ok) META = await metaRes.json();
  } catch (e) {}

  try {
    const perfRes = await fetch('/api/results');
    if (perfRes.ok) RES = await perfRes.json();
  } catch (e) {}

  buildLabels();
  bindMetadata();
  buildForm();
  handleRouting();
  executePrediction();
  icons();
})();
})();
