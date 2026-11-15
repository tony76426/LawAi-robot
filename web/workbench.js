(function () {
  'use strict';

  const byId = (id) => document.getElementById(id);
  const introElement = byId('intro');
  const modes = {
    chat: {
      label: '智能問答', eyebrow: 'LEGAL INTELLIGENCE / 01',
      title: '把法律問題，整理成可理解的方向。',
      subtitle: '先從知識庫取得相近問題，再確認哪些內容需要進一步查證。',
      intro: '從一個問題開始。',
      description: '你可以直接輸入遇到的法律狀況，或從下方選一個常見問題。我們會標示資料來源與目前檢索模式，不把不確定的資訊當成結論。',
      evidence: '智能問答：以 FAQ 知識庫為基礎，不直接生成未查核的法條或判決。',
      button: '送出問題',
      samples: ['我要離婚，應該先準備什麼？', '房東不退押金怎麼辦？', '民事訴訟大概要多久？', '我要找律師，收費如何計算？']
    },
    search: {
      label: '語意檢索', eyebrow: 'KNOWLEDGE RETRIEVAL / 02',
      title: '每一筆搜尋結果，都有跡可循。',
      subtitle: '查看候選資料、來源類型與匹配分數，不把搜尋排名當作法律結論。',
      intro: '搜尋知識，而不只是答案。',
      description: '輸入你想查找的問題，我們將列出最相關的資料。若真實向量索引尚未建立，會清楚改用關鍵字模式。',
      evidence: '資料檢索：取得 Top 5 候選，區分 FAQ、法律條文與裁判來源；分數不是法律正確率。',
      button: '搜尋資料',
      samples: ['租客不繳租金也不搬', '我父親過世怎麼繼承房產？', '沒有借據能告嗎？', '對方不給扶養費怎麼辦？']
    },
    compare: {
      label: '模型觀點對照', eyebrow: 'MODEL COMPARISON / 03',
      title: '不同觀點，交叉檢查論證。',
      subtitle: '支持方、反方與審閱者在相同來源下分析，分歧及限制都應被看見。',
      intro: '讓不同模型，檢視同一個爭點。',
      description: '此功能只在後端已建立真實向量索引、具有人工審核法律來源、開啟比較功能且完成 Token 驗證後才能使用。未符合條件時不會產生模型費用。',
      evidence: '模型對照：模型 A 獨立提出支持論點、模型 B 提出反方抗辯、第三模型檢視兩者；不能取代律師核實。',
      button: '開始對照分析',
      samples: ['借款只有匯款紀錄沒有借據，可以請求返還嗎？', '租約提前終止，押金可以全部沒收嗎？', '離婚後房產如何分配？', '車禍沒有直接碰撞，是否仍有責任？']
    }
  };

  const state = {
    mode: 'chat', busy: false, token: '', health: null, connected: false,
    apiBase: '', messages: {chat: [], search: [], compare: []},
    hits: {chat: [], search: [], compare: []},
    recognition: null, listening: false, toastTimer: null
  };

  const node = (tag, cls, text) => {
    const el = document.createElement(tag);
    if (cls) el.className = cls;
    if (text !== undefined && text !== null) el.textContent = String(text);
    return el;
  };

  function resolveApiBase() {
    const configured = String(window.LAWAI_API_BASE_URL || '').trim();
    if (!configured) return '';
    try {
      const url = new URL(configured);
      const local = ['localhost', '127.0.0.1', '[::1]'].includes(url.hostname);
      if (url.protocol !== 'https:' && !(local && url.protocol === 'http:')) {
        throw new Error('API 需要 HTTPS（或 localhost 的 HTTP）。');
      }
      if (url.username || url.password || url.search || url.hash) {
        throw new Error('API 位址不得包含帳密、查詢參數或片段。');
      }
      return url.origin + (url.pathname === '/' ? '' : url.pathname.replace(/\/+$/, ''));
    } catch (e) {
      return null;
    }
  }

  function endpoint(path) {
    return (state.apiBase || '') + path;
  }

  function notify(message) {
    const toast = byId('toast');
    toast.textContent = message;
    toast.hidden = false;
    if (state.toastTimer) clearTimeout(state.toastTimer);
    state.toastTimer = setTimeout(() => { toast.hidden = true; }, 5200);
  }

  async function callApi(path, payload, timeoutMs) {
    if (state.apiBase === null) throw new Error('API 網址格式不安全，請檢查 web/config.js。');
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs || 35000);
    const headers = {'Accept': 'application/json'};
    if (payload !== undefined) headers['Content-Type'] = 'application/json';
    if (state.token) headers['Authorization'] = 'Bearer ' + state.token;
    try {
      const response = await fetch(endpoint(path), {
        method: payload === undefined ? 'GET' : 'POST',
        mode: 'cors', credentials: 'omit', cache: 'no-store',
        headers, signal: controller.signal,
        body: payload === undefined ? undefined : JSON.stringify(payload)
      });
      const text = await response.text();
      let data;
      try { data = JSON.parse(text); } catch (_) { data = null; }
      if (!response.ok) {
        const error = (data && typeof data.error === 'string' && data.error) ||
          '伺服器回應狀態 ' + response.status;
        throw new Error(error);
      }
      if (!data || typeof data !== 'object') throw new Error('後端回傳的格式不是有效 JSON。');
      return data;
    } catch (error) {
      if (error.name === 'AbortError') throw new Error('請求逾時；如是付費模型比較，先確認後端狀態，避免重複送出。');
      if (error instanceof TypeError) throw new Error('無法連線到 API。請確認 Render 服務是否啟用，以及跨網域 CORS 設定。');
      throw error;
    } finally {
      clearTimeout(timer);
    }
  }

  function updateConnection() {
    const dotClass = 'status-dot ' + (state.connected ? 'online' : 'offline');
    byId('sidebarDot').className = dotClass;
    byId('topbarDot').className = dotClass;
    const h = state.health || {};
    byId('connectionLabel').textContent = state.connected ? '已連線' : '尚未連線';
    byId('sidebarConnection').textContent = state.connected ? '後端連線正常' : '後端尚未連線';
    byId('sidebarFaqCount').textContent = state.connected ? String(h.records || 0) + ' 筆' : '—';
    byId('sidebarLegalCount').textContent = state.connected ? String(h.approved_legal_sources || 0) + ' 筆' : '—';
    const index = h.index_status;
    byId('sidebarIndexStatus').textContent = !state.connected ? '待連線' :
      index === 'ready' ? '已就緒' : '尚未就緒';
    refreshComposer();
  }

  async function checkHealth() {
    try {
      const health = await callApi('/api/health', undefined, 9000);
      if (!health.ok) throw new Error('健康檢查沒有通過。');
      state.health = health;
      state.connected = true;
      if (state.mode === 'compare') refreshEvidence('比較模式需要核對索引與法律來源。', []);
      else refreshEvidence(
        health.index_status === 'ready'
          ? '向量索引已就緒。請輸入問題開始檢索。'
          : '尚未建立有效向量索引；目前會使用關鍵字檢索。', []);
    } catch (error) {
      state.health = null;
      state.connected = false;
      refreshEvidence('後端尚未連線。' + error.message, [], true);
    }
    updateConnection();
  }

  function compareReason() {
    const h = state.health;
    if (!state.connected || !h) return '後端尚未連線；目前 Render 服務可能仍暫停。';
    if (h.index_status !== 'ready') return '尚未建立真實的 Embedding 索引。';
    if (!Number(h.approved_legal_sources)) return '尚未匯入經人工審核的法律條文或裁判來源。';
    if (!h.comparison_enabled) return '後端尚未開啟模型比較功能。';
    if (!state.token) return '請於「連線設定」輸入研究者 API Token。';
    return '';
  }

  function refreshComposer() {
    const reason = compareReason();
    const unavailable = state.mode === 'compare' && Boolean(reason);
    const button = byId('sendButton');
    button.disabled = state.busy || unavailable;
    button.textContent = state.busy ? '處理中…' : modes[state.mode].button;
    if (!state.busy) button.append(node('span', '', '↗'));
    byId('questionInput').disabled = state.busy;
    byId('micButton').disabled = state.busy;
    byId('composerHint').textContent = unavailable ? reason :
      state.busy ? '正在處理請求，請勿重複送出。' : 'Enter 送出 · Shift + Enter 換行';
  }

  function safeSourceUrl(raw) {
    if (typeof raw !== 'string') return null;
    try {
      const url = new URL(raw);
      return url.protocol === 'https:' && !url.username && !url.password ? url.href : null;
    } catch (_) { return null; }
  }

  function sourceCard(hit, position) {
    const el = node('article', 'source-card');
    const kind = hit.kind === 'law' ? '法律條文' :
      hit.kind === 'judgment' ? '法院裁判' : '網站 FAQ';
    el.append(node('div', 'source-kicker', kind + ' · ' + String(position + 1).padStart(2, '0')));
    el.append(node('div', 'source-title', hit.question || hit.title || '未命名來源'));
    if (typeof hit.snippet === 'string' && hit.snippet) {
      el.append(node('p', 'source-excerpt', hit.snippet.slice(0, 250)));
    }
    const isLegal = hit.kind === 'law' || hit.kind === 'judgment';
    if (isLegal && !hit.approved_legal) {
      el.append(node('div', 'source-kicker', '尚未完成法律來源人工審核'));
    }
    if (typeof hit.score === 'number') {
      const score = Math.min(100, Math.max(0, hit.score * 100));
      const scoreRow = node('div', 'source-score');
      scoreRow.append(node('span', '', '匹配 ' + score.toFixed(1) + '%'));
      const meter = node('span', 'source-meter');
      const fill = node('span');
      fill.style.width = score + '%';
      meter.append(fill);
      scoreRow.append(meter);
      el.append(scoreRow);
    }
    const sourceUrl = safeSourceUrl(hit.source_url);
    if (sourceUrl) {
      const link = node('a', '', '開啟來源原文 ↗');
      link.href = sourceUrl;
      link.target = '_blank';
      link.rel = 'noopener noreferrer';
      el.append(link);
    }
    return el;
  }

  function refreshEvidence(status, hits, error) {
    const statusEl = byId('evidenceStatus');
    statusEl.textContent = status;
    statusEl.className = 'evidence-status' + (error ? ' error' : '');
    const list = byId('sourcesList');
    list.replaceChildren();
    const rows = Array.isArray(hits) ? hits : [];
    byId('sourceCount').textContent = String(rows.length);
    if (!rows.length) {
      list.append(node('p', 'no-sources', '尚無可顯示的來源。資料庫不足時，不會捏造法條或裁判。'));
    } else {
      rows.forEach((hit, i) => list.append(sourceCard(hit, i)));
    }
  }

  function renderTextMessage(message) {
    const row = node('article', 'message-row ' + (message.role === 'user' ? 'user' : 'assistant'));
    if (message.role !== 'user') row.append(node('div', 'message-icon', '⚖'));
    const block = node('div', 'message-block');
    block.append(node('div', 'message-author', message.role === 'user' ? '您' : 'LawAI 研究助理'));
    block.append(node('div', 'message-body', message.text));
    if (message.meta) block.append(node('div', 'message-foot', message.meta));
    if (Array.isArray(message.hits) && message.hits.length) {
      const list = node('div', 'inline-source-list');
      message.hits.forEach((hit, i) => list.append(sourceCard(hit, i)));
      block.append(list);
    }
    if (message.analyses) {
      const grid = node('div', 'analysis-grid');
      [['support','模型 A｜支持論點'], ['challenge','模型 B｜反方抗辯'],
       ['review','獨立審閱｜分歧與風險']].forEach(([key, title]) => {
        const result = message.analyses[key] || {};
        const card = node('section', 'analysis-card ' + (key === 'review' ? 'review' : ''));
        card.append(node('h3', '', title));
        card.append(node('p', '', result.text || '沒有可用內容。'));
        const audit = result.citation_ids || {};
        const invalid = Array.isArray(audit.invalid) ? audit.invalid : [];
        if (invalid.length) {
          card.append(node('div', 'comparison-meta', '來源 ID 未在本次資料包中：' + invalid.join(', ')));
        }
        grid.append(card);
      });
      block.append(grid);
      block.append(node('div', 'comparison-meta', '上述內容為模型產生的研究草稿；法律效力、原文與引證適用性仍須律師核查。'));
    }
    row.append(block);
    return row;
  }

  function scrollToEnd() {
    const panel = byId('conversation');
    panel.scrollTop = panel.scrollHeight;
  }

  function renderThread() {
    const panel = byId('conversation');
    panel.replaceChildren();
    const messages = state.messages[state.mode];
    if (!messages.length) {
      if (introElement) panel.append(introElement);
    } else {
      messages.forEach((message) => panel.append(renderTextMessage(message)));
    }
    scrollToEnd();
    const hits = state.hits[state.mode];
    if (hits.length) refreshEvidence('已取得 ' + hits.length + ' 筆檢索候選，請核對來源與內容。', hits);
    else refreshEvidence('此模式尚未檢索到可顯示的來源。', []);
  }

  function renderIntro() {
    const intro = introElement;
    if (!intro) return;
    const mode = modes[state.mode];
    intro.querySelector('#introTitle').textContent = mode.intro;
    intro.querySelector('#introDescription').textContent = mode.description;
    const list = intro.querySelector('#sampleList');
    list.replaceChildren();
    mode.samples.forEach((text) => {
      const button = node('button', 'sample-btn');
      button.type = 'button';
      button.append(node('span', '', text), node('span', '', '↗'));
      button.addEventListener('click', () => {
        byId('questionInput').value = text;
        autosizeTextarea();
        byId('questionInput').focus();
      });
      list.append(button);
    });
  }

  function switchMode(mode) {
    if (!modes[mode] || state.busy) return;
    state.mode = mode;
    document.querySelectorAll('[data-mode]').forEach((button) => {
      const active = button.dataset.mode === mode;
      button.classList.toggle('active', active);
      if (active) button.setAttribute('aria-current', 'page');
      else button.removeAttribute('aria-current');
    });
    const info = modes[mode];
    byId('pageEyebrow').textContent = info.eyebrow;
    byId('pageTitle').textContent = info.title;
    byId('pageSubtitle').textContent = info.subtitle;
    byId('modeDescription').textContent = info.evidence;
    byId('questionInput').placeholder =
      mode === 'compare' ? '描述需要比較的法律爭點……' :
      mode === 'search' ? '輸入你想查找的法律問題……' :
      '請輸入你的法律問題，越具體越容易找到相關資料……';
    renderIntro();
    renderThread();
    refreshComposer();
    closeSidebar();
    if (mode === 'compare') {
      const reason = compareReason();
      if (reason) refreshEvidence('模型比較尚不可用：' + reason, [], true);
    }
  }

  function addMessage(message) {
    state.messages[state.mode].push(message);
    const panel = byId('conversation');
    if (byId('intro')) byId('intro').remove();
    panel.append(renderTextMessage(message));
    scrollToEnd();
  }

  function loadingRow() {
    const row = node('div', 'message-row assistant');
    row.id = 'pendingRow';
    row.append(node('div', 'message-icon', '⚖'));
    const block = node('div', 'message-block');
    block.append(node('div', 'message-author', 'LawAI 研究助理'));
    const p = node('div', 'pending-line');
    p.append(node('span', '', '正在檢索資料'));
    for (let i = 0; i < 3; i += 1) p.append(node('span', 'pulse-dot'));
    block.append(p);
    row.append(block);
    byId('conversation').append(row);
    scrollToEnd();
  }

  function removeLoading() {
    const pending = byId('pendingRow');
    if (pending) pending.remove();
  }

  function autosizeTextarea() {
    const textarea = byId('questionInput');
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(200, Math.max(49, textarea.scrollHeight)) + 'px';
  }

  async function sendQuestion(event) {
    if (event) event.preventDefault();
    if (state.busy) return;
    const input = byId('questionInput');
    const question = input.value.trim();
    if (!question) { input.focus(); return; }
    if (question.length > 2000) { notify('問題最多 2000 字。'); return; }
    if (state.mode === 'compare') {
      const reason = compareReason();
      if (reason) { notify(reason); return; }
    }
    state.busy = true;
    refreshComposer();
    stopRecognition();
    input.value = '';
    autosizeTextarea();
    const mode = state.mode;
    addMessage({role:'user', text:question});
    loadingRow();
    try {
      let response, hits, body, meta;
      if (mode === 'search') {
        response = await callApi('/api/search', {question}, 35000);
        hits = Array.isArray(response.hits) ? response.hits.slice(0, 5) : [];
        body = hits.length
          ? '找到 ' + hits.length + ' 筆相近資料。請依來源類型、實際內容及原文連結判斷相關性。'
          : '目前沒有達到匹配門檻的資料，不會強行回傳無關的法律答案。';
        meta = response.mode === 'hybrid' ? '向量＋關鍵字混合檢索' : '關鍵字檢索';
      } else if (mode === 'compare') {
        response = await callApi('/api/compare', {question}, 100000);
        hits = (response.sources || []).map((source) => ({
          question: source.title, kind: source.kind, source_url: source.source_url,
          approved_legal: true
        }));
        body = response.status === 'completed'
          ? '已完成不同觀點的初步比較。三份模型內容應逐項對照原始法規與裁判，不能視為法律定論。'
          : response.answer || '法律來源不足，未啟動模型分析。';
        meta = response.status === 'completed' ? '模型對照完成 · 仍需人工審核' : '來源不足 · 未執行模型對照';
      } else {
        response = await callApi('/api/generate', {question}, 35000);
        hits = Array.isArray(response.sources) ? response.sources.slice(0, 5) : [];
        body = response.answer || '目前沒有可用的參考答案。';
        meta = response.mode === 'hybrid' ? '向量＋關鍵字檢索 · FAQ 參考'
          : '關鍵字檢索 · FAQ 參考';
      }
      state.hits[mode] = hits;
      removeLoading();
      const message = {role:'assistant', text:body, meta, hits: mode === 'search' ? hits : [],
        analyses: mode === 'compare' && response.status === 'completed' ? response.analyses : null};
      addMessage(message);
      refreshEvidence('已完成本次檢索。' + (response.warning ? ' 向量服務暫時不可用，請注意降級狀態。' : ''), hits);
    } catch (error) {
      removeLoading();
      state.hits[mode] = [];
      addMessage({role:'assistant', text:'這次請求沒有成功完成。\n\n' +
        error.message + '\n\n請先檢查連線與服務狀態；模型比較若已逾時，不要立即重複送出。'});
      refreshEvidence('請求失敗：' + error.message, [], true);
    } finally {
      state.busy = false;
      refreshComposer();
      input.focus();
    }
  }

  function closeSidebar() {
    byId('sidebar').classList.remove('open');
    byId('sidebarScrim').hidden = true;
  }
  function toggleSidebar() {
    const open = !byId('sidebar').classList.contains('open');
    byId('sidebar').classList.toggle('open', open);
    byId('sidebarScrim').hidden = !open;
  }

  function stopRecognition() {
    if (state.recognition) {
      try { state.recognition.stop(); } catch (_) {}
      state.recognition = null;
    }
    state.listening = false;
    byId('micButton').classList.remove('listening');
    byId('micButton').setAttribute('aria-label','開始語音輸入');
  }

  function speechInput() {
    if (state.listening) { stopRecognition(); return; }
    const Speech = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!Speech) { notify('目前瀏覽器不支援語音辨識，請改用文字輸入。'); return; }
    if (!window.isSecureContext) { notify('語音輸入需要 HTTPS 或 localhost。'); return; }
    const recognizer = new Speech();
    recognizer.lang = 'zh-TW';
    recognizer.interimResults = false;
    recognizer.continuous = false;
    recognizer.onresult = (event) => {
      const transcript = event.results && event.results[0] && event.results[0][0] &&
        event.results[0][0].transcript;
      if (transcript) {
        const input = byId('questionInput');
        input.value += (input.value.trim() ? '\n' : '') + transcript;
        autosizeTextarea();
        input.focus();
      }
    };
    recognizer.onend = () => { state.recognition = null;state.listening = false;
      byId('micButton').classList.remove('listening'); };
    recognizer.onerror = () => { notify('語音辨識未完成，請檢查麥克風權限。');stopRecognition(); };
    state.recognition = recognizer;
    state.listening = true;
    byId('micButton').classList.add('listening');
    byId('micButton').setAttribute('aria-label','停止語音輸入');
    try { recognizer.start(); } catch (_) { stopRecognition(); notify('無法啟動語音辨識。'); }
  }

  function bindEvents() {
    byId('questionForm').addEventListener('submit', sendQuestion);
    byId('questionInput').addEventListener('input', autosizeTextarea);
    byId('questionInput').addEventListener('keydown', (event) => {
      if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
        event.preventDefault();
        if (!byId('sendButton').disabled) sendQuestion(event);
      }
    });
    byId('micButton').addEventListener('click', speechInput);
    byId('modeNav').addEventListener('click', (event) => {
      const selected = event.target.closest('[data-mode]');
      if (selected) switchMode(selected.dataset.mode);
    });
    byId('newConversation').addEventListener('click', () => {
      if (state.busy) return;
      state.messages[state.mode] = [];
      state.hits[state.mode] = [];
      byId('questionInput').value = '';
      autosizeTextarea();
      renderThread();
      byId('questionInput').focus();
    });
    byId('sidebarToggle').addEventListener('click', toggleSidebar);
    byId('sidebarClose').addEventListener('click', closeSidebar);
    byId('sidebarScrim').addEventListener('click', closeSidebar);
    byId('settingsOpen').addEventListener('click', () => {
      byId('settingsApiBase').textContent = state.apiBase === null ? '設定值無效' :
        (state.apiBase || (location.origin === 'null' ? '請在 config.js 設定 API 位址' : location.origin + '（同網域）'));
      byId('apiToken').value = state.token;
      byId('settingsDialog').showModal();
      checkHealth();
    });
    byId('settingsClose').addEventListener('click', () => byId('settingsDialog').close());
    byId('saveSettings').addEventListener('click', () => {
      state.token = byId('apiToken').value.trim();
      byId('settingsDialog').close();
      refreshComposer();
      notify('本次網頁的連線設定已更新；Token 不會持久保存。');
    });
    byId('forgetToken').addEventListener('click', () => {
      state.token = '';
      byId('apiToken').value = '';
      refreshComposer();
      notify('本次網頁記憶體中的 Token 已清除。');
    });
    byId('settingsDialog').addEventListener('close', () => {
      byId('apiToken').value = '';
    });
  }

  function start() {
    state.apiBase = resolveApiBase();
    bindEvents();
    renderIntro();
    switchMode('chat');
    checkHealth();
  }

  start();
}());
