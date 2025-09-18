async function uploadCV(e) {
	e.preventDefault();
	const fileInput = document.getElementById('cvFile');
	if (!fileInput.files.length) return;
	const formData = new FormData();
	formData.append('file', fileInput.files[0]);
	const res = await fetch('/api/upload', { method: 'POST', body: formData });
	const data = await res.json();
	document.getElementById('uploadResult').textContent = JSON.stringify(data, null, 2);
}

async function runSearch() {
	const name = document.getElementById('searchName').value;
	const email = document.getElementById('searchEmail').value;
	const minExp = document.getElementById('searchExp').value;
	const text = document.getElementById('searchText').value;
	const params = new URLSearchParams();
	if (name) params.append('name', name);
	if (email) params.append('email', email);
	if (minExp) params.append('min_experience', minExp);
	if (text) params.append('text', text);
	const res = await fetch('/api/search?' + params.toString());
	const data = await res.json();
	const list = document.getElementById('searchResults');
	list.innerHTML = '';
	(data.results || []).forEach(r => {
		const li = document.createElement('li');
		li.textContent = `${r.name || 'Unknown'} | ${r.email || ''} | ${r.total_experience_years || 0} yrs`;
		list.appendChild(li);
	});
}

async function sendChat() {
	const input = document.getElementById('chatInput');
	const msg = input.value.trim();
	if (!msg) return;
	appendChat('You', msg);
	input.value = '';
	const res = await fetch('/api/chat', {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({ message: msg })
	});
	const data = await res.json();
	appendChat('AI', data.reply || '');
}

function appendChat(sender, text) {
	const box = document.getElementById('chatBox');
	const div = document.createElement('div');
	div.className = 'chat-msg';
	div.innerHTML = `<span class="chat-user">${sender}:</span> ${escapeHtml(text)}`;
	box.appendChild(div);
	box.scrollTop = box.scrollHeight;
}

function escapeHtml(s) {
	return s.replace(/[&<>"] /g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',' ':' '}[c]));
}

// Voice search using Web Speech API
let recognition = null;
if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
	const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
	recognition = new SR();
	recognition.lang = 'en-US';
	recognition.interimResults = false;
	recognition.maxAlternatives = 1;
}

function startVoiceSearch() {
	if (!recognition) {
		alert('Speech recognition not supported in this browser.');
		return;
	}
	recognition.onresult = (event) => {
		const transcript = event.results[0][0].transcript;
		document.getElementById('searchText').value = transcript;
		runSearch();
	};
	recognition.start();
}

document.getElementById('uploadForm').addEventListener('submit', uploadCV);

document.getElementById('searchBtn').addEventListener('click', runSearch);

document.getElementById('chatSend').addEventListener('click', sendChat);

document.getElementById('voiceBtn').addEventListener('click', startVoiceSearch);