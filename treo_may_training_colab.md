/**
 * =========================================================================
 * GOOGLE COLAB SMART AUTO-KEEP-ALIVE & ANTI-TIMEOUT TOOL
 * Dành cho các tác vụ Train Deep Learning dài (> 3 - 12 tiếng)
 * =========================================================================
 */
(() => {
    // Nếu đã có phiên chạy trước đó thì xóa trước khi bật mới
    if (window._colabKeepAliveInterval) {
        clearInterval(window._colabKeepAliveInterval);
    }
    if (window._colabAudioContext) {
        window._colabAudioContext.close();
    }

    const CHECK_INTERVAL_SEC = 60; // Chu kỳ nhấp kiểm tra (60 giây/lần)
    let cycleCount = 0;
    const startTime = new Date();

    // 1. Chống đóng băng Tab ngầm (Anti-Tab-Sleeping via Web Audio)
    let audioCtx = null;
    try {
        const AudioContext = window.AudioContext || window.webkitAudioContext;
        if (AudioContext) {
            audioCtx = new AudioContext();
            const osc = audioCtx.createOscillator();
            const gain = audioCtx.createGain();
            gain.gain.value = 0.00001; // Âm lượng thực tế = 0 (không phát ra tiếng)
            osc.connect(gain);
            gain.connect(audioCtx.destination);
            osc.start();
            window._colabAudioContext = audioCtx;
        }
    } catch (e) {
        console.warn("AudioContext không khả dụng, bỏ qua bước chống sleep âm thanh.");
    }

    // 2. Hàm click thông minh xử lý cả Shadow DOM & Dialog Popups
    function keepAliveAction() {
        cycleCount++;
        const now = new Date();
        const elapsedMin = Math.round((now - startTime) / 60000);
        let actionTaken = [];

        // A. Tự động nhấn nút Reconnect / OK nếu xuất hiện bảng cảnh báo
        const okButtons = [
            document.querySelector("#ok"),
            document.querySelector("colab-dialog mwc-button#ok"),
            document.querySelector("mwc-button#ok"),
            document.querySelector("colab-dialog")?.shadowRoot?.querySelector("#ok")
        ];
        for (const btn of okButtons) {
            if (btn && btn.offsetParent !== null) {
                btn.click();
                actionTaken.push("Đã xác nhận Popup Reconnect/OK");
            }
        }

        // B. Tương tác với nút Connect (hỗ trợ cả cấu trúc chuẩn và Shadow DOM)
        const colabButton = document.querySelector("colab-connect-button");
        if (colabButton) {
            const connectBtn = colabButton.shadowRoot ? colabButton.shadowRoot.querySelector("#connect") : colabButton.querySelector("#connect");
            if (connectBtn) {
                connectBtn.click();
                actionTaken.push("Đã click nút Connect");
            } else {
                colabButton.click();
                actionTaken.push("Đã click colab-connect-button");
            }
        } else {
            // Dự phòng: tương tác vùng notebook nhẹ để tạo event người dùng
            window.dispatchEvent(new MouseEvent('mousemove', { bubbles: true }));
            actionTaken.push("Đã kích hoạt Mouse Event");
        }

        // C. Giữ cho Web Audio Context luôn Running
        if (audioCtx && audioCtx.state === 'suspended') {
            audioCtx.resume();
        }

        const logMsg = actionTaken.join(" + ");
        console.log(`[🚀 Colab Keep-Alive | Chu kỳ #${cycleCount}] ${now.toLocaleTimeString()} (Đã chạy: ${elapsedMin} phút) -> ${logMsg} ✓`);
    }

    // Chạy lần đầu ngay lập tức, sau đó lặp lại mỗi 60 giây
    keepAliveAction();
    window._colabKeepAliveInterval = setInterval(keepAliveAction, CHECK_INTERVAL_SEC * 1000);

    // Cung cấp hàm dừng tiện lợi
    window.stopColabKeepAlive = () => {
        clearInterval(window._colabKeepAliveInterval);
        if (window._colabAudioContext) window._colabAudioContext.close();
        console.log("🛑 [Colab Keep-Alive] Đã dừng công cụ tự động.");
    };

    console.log("%c🔥 [Colab Keep-Alive Đã BẬT] Tự động nhấp mỗi 60s. Gõ stopColabKeepAlive() để tắt.", "color: #00FF66; font-size: 13px; font-weight: bold;");
})();
