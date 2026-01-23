// Global state
let detectionActive = true;
let stats = {
    faces: 0,
    humans: 0,
    vehicles: 0,
    cars: 0,
    motorcycles: 0,
    buses: 0,
    trucks: 0,
    traffic_lights: 0,
    dogs: 0,
    cats: 0,
    cows: 0,
    horses: 0,
    zebra_crossings: 0,
    footpaths: 0,
    buffaloes: 0,
    bullock_carts: 0,
    fps: 0
};

// Handle navigation
document.addEventListener('DOMContentLoaded', function () {
    initializeNavigation();
    initializeDetectionControls();
    initializeFullscreen();
    startStatsPolling();
    initializeVideoStream();
    initializeTiltEffects();
});

function initializeTiltEffects() {
    const cards = document.querySelectorAll('.tilt-card');

    cards.forEach(card => {
        card.addEventListener('mousemove', (e) => {
            const rect = card.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            const centerX = rect.width / 2;
            const centerY = rect.height / 2;

            const rotateX = ((y - centerY) / centerY) * -10; // Max rotation 10deg
            const rotateY = ((x - centerX) / centerX) * 10;

            card.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) scale(1.05)`;
        });

        card.addEventListener('mouseleave', () => {
            card.style.transform = 'perspective(1000px) rotateX(0) rotateY(0) scale(1)';
        });
    });
}

function initializeNavigation() {
    const sections = document.querySelectorAll('.section');
    const navLinks = document.querySelectorAll('.nav-link');

    // Get section from URL or default to dashboard
    const currentPath = window.location.pathname;
    const urlParams = new URLSearchParams(window.location.search);
    let activeSection = 'dashboard';

    if (urlParams.has('section')) {
        activeSection = urlParams.get('section');
    } else if (currentPath.includes('/team')) {
        activeSection = 'team';
    } else if (currentPath.includes('/about')) {
        activeSection = 'about';
    }

    // Show appropriate section
    sections.forEach(sec => {
        sec.classList.remove('active');
        if (sec.id === activeSection) {
            sec.classList.add('active');
        }
    });

    // Update nav links
    navLinks.forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('data-section') === activeSection) {
            link.classList.add('active');
        }
    });

    // Handle nav link clicks
    navLinks.forEach(link => {
        link.addEventListener('click', function (e) {
            const targetSection = link.getAttribute('data-section');

            // If it's a real link (no data-section), let the browser handle it
            if (!targetSection) return;

            e.preventDefault();

            // Update URL without reload
            const newPath = targetSection === 'dashboard' ? '/' : `/${targetSection}`;
            window.history.pushState({}, '', newPath);

            // Update sections
            sections.forEach(sec => {
                sec.classList.remove('active');
                if (sec.id === targetSection) {
                    sec.classList.add('active');
                }
            });

            // Update nav links
            navLinks.forEach(l => l.classList.remove('active'));
            link.classList.add('active');
        });
    });
}

function initializeDetectionControls() {
    const toggleBtn = document.getElementById('toggle-btn');
    const statusDot = document.getElementById('status-dot');
    const statusText = document.getElementById('status-text');

    if (toggleBtn) {
        toggleBtn.addEventListener('click', async function () {
            toggleBtn.disabled = true;
            toggleBtn.style.opacity = '0.6';

            try {
                const response = await fetch('/toggle');
                const data = await response.json();
                detectionActive = data.status;

                updateDetectionUI(detectionActive);
            } catch (error) {
                console.error('Error toggling detection:', error);
                showNotification('Error toggling detection', 'error');
            } finally {
                toggleBtn.disabled = false;
                toggleBtn.style.opacity = '1';
            }
        });
    }

    // Initial status check and Auto-Start
    updateDetectionStatus().then(() => {
        if (!detectionActive) {
            // Auto-start detection if not active
            console.log("Auto-starting detection...");
            const toggleBtn = document.getElementById('toggle-btn');
            if (toggleBtn) toggleBtn.click(); // Simulate click to trigger server toggle
        }
    });
}

function updateDetectionUI(active) {
    const toggleBtn = document.getElementById('toggle-btn');
    const statusDot = document.getElementById('status-dot');
    const statusText = document.getElementById('status-text');

    if (toggleBtn) {
        if (active) {
            toggleBtn.classList.remove('inactive');
            toggleBtn.innerHTML = '<span class="btn-icon">🔍</span><span class="btn-text">Disable Detection</span>';
        } else {
            toggleBtn.classList.add('inactive');
            toggleBtn.innerHTML = '<span class="btn-icon">⏸</span><span class="btn-text">Enable Detection</span>';
        }
    }

    if (statusDot) {
        if (active) {
            statusDot.classList.remove('inactive');
            statusText.textContent = 'Active';
        } else {
            statusDot.classList.add('inactive');
            statusText.textContent = 'Inactive';
        }
    }
}

async function updateDetectionStatus() {
    try {
        const response = await fetch('/status');
        const data = await response.json();
        detectionActive = data.status;
        updateDetectionUI(detectionActive);
    } catch (error) {
        console.error('Error fetching status:', error);
    }
}

function initializeFullscreen() {
    const fullscreenBtn = document.getElementById('fullscreen-btn');
    const videoContainer = document.querySelector('.video-container');

    if (fullscreenBtn && videoContainer) {
        fullscreenBtn.addEventListener('click', function () {
            if (!document.fullscreenElement) {
                videoContainer.requestFullscreen().catch(err => {
                    console.error('Error entering fullscreen:', err);
                });
            } else {
                document.exitFullscreen();
            }
        });
    }
}

function startStatsPolling() {
    // Poll status every 2 seconds
    setInterval(updateDetectionStatus, 2000);

    // Poll stats every second
    setInterval(updateStatsFromAPI, 1000);
    updateStatsFromAPI(); // Initial call
}

async function updateStatsFromAPI() {
    try {
        const response = await fetch('/stats');
        const data = await response.json();
        stats.faces = data.faces || 0;
        stats.humans = data.humans || 0;
        stats.vehicles = data.vehicles || 0;
        stats.cars = data.cars || 0;
        stats.motorcycles = data.motorcycles || 0;
        stats.buses = data.buses || 0;
        stats.trucks = data.trucks || 0;
        stats.traffic_lights = data.traffic_lights || 0;
        stats.dogs = data.dogs || 0;
        stats.cats = data.cats || 0;
        stats.cows = data.cows || 0;
        stats.horses = data.horses || 0;
        stats.zebra_crossings = data.zebra_crossings || 0;
        stats.footpaths = data.footpaths || 0;
        stats.buffaloes = data.buffaloes || 0;
        stats.bullock_carts = data.bullock_carts || 0;
        stats.fps = data.fps || 0;
        updateStatsDisplay();
    } catch (error) {
        console.error('Error fetching stats:', error);
    }
}

function updateStatsDisplay() {
    // Update all stat displays
    const statElements = {
        'face-count': stats.faces,
        'human-count': stats.humans,
        'vehicle-count': stats.vehicles,
        'traffic-light-count': stats.traffic_lights,
        'dog-count': stats.dogs,
        'cat-count': stats.cats,
        'cow-count': stats.cows,
        'zebra-crossing-count': stats.zebra_crossings,
        'buffalo-count': stats.buffaloes,
        'bullock-cart-count': stats.bullock_carts,
        'fps-value': stats.fps
    };

    for (const [id, value] of Object.entries(statElements)) {
        const element = document.getElementById(id);
        if (element) {
            const isFloat = id === 'fps-value';
            const currentValue = isFloat ? parseFloat(element.textContent) || 0 : parseInt(element.textContent) || 0;
            animateValue(element, currentValue, value, isFloat);
        }
    }
}

function animateValue(element, start, end, isFloat = false) {
    if (start === end) return;

    const duration = 500;
    const startTime = performance.now();
    const difference = end - start;

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);

        // Easing function
        const easeOutQuart = 1 - Math.pow(1 - progress, 4);
        const current = start + (difference * easeOutQuart);

        if (isFloat) {
            element.textContent = current.toFixed(1);
        } else {
            element.textContent = Math.round(current);
        }

        if (progress < 1) {
            requestAnimationFrame(update);
        } else {
            if (isFloat) {
                element.textContent = end.toFixed(1);
            } else {
                element.textContent = end;
            }
        }
    }

    requestAnimationFrame(update);
}

function initializeVideoStream() {
    const videoStream = document.getElementById('video-stream');

    if (videoStream) {
        // Handle video load
        videoStream.addEventListener('load', function () {
            hideLoading();
        });

        // Handle video errors
        videoStream.addEventListener('error', function () {
            showError();
        });

        // Try to extract stats from video (if text overlay is visible)
        // This is a placeholder - in production, you'd want an API endpoint
        setInterval(() => {
            // Simulate or extract stats from video
            // For now, we'll just keep the display updated
        }, 1000);
    }
}

function hideLoading() {
    const overlay = document.getElementById('video-overlay');
    if (overlay) {
        overlay.classList.add('hidden');
    }
}

function showError() {
    const overlay = document.getElementById('video-overlay');
    if (overlay) {
        overlay.innerHTML = `
            <div style="text-align: center; color: white;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">⚠️</div>
                <p>Camera feed unavailable</p>
                <p style="font-size: 0.9rem; opacity: 0.8; margin-top: 0.5rem;">Please check your camera connection</p>
            </div>
        `;
        overlay.classList.remove('hidden');
    }
}

function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: ${type === 'error' ? '#ef4444' : '#10b981'};
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        z-index: 1000;
        animation: slideInRight 0.3s ease;
    `;
    notification.textContent = message;

    document.body.appendChild(notification);

    setTimeout(() => {
        notification.style.animation = 'slideOutRight 0.3s ease';
        setTimeout(() => {
            document.body.removeChild(notification);
        }, 300);
    }, 3000);
}

// Add CSS animations for notifications
const style = document.createElement('style');
style.textContent = `
    @keyframes slideInRight {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    @keyframes slideOutRight {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);