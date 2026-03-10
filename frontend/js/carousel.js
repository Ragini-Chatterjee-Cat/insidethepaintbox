// Artwork Carousel - Filmstrip Style (all visible, one in focus)
document.addEventListener('DOMContentLoaded', function() {
    const carousel = document.querySelector('.artwork-carousel');
    if (!carousel) return;

    const container = carousel.querySelector('.carousel-container');
    const track = carousel.querySelector('.carousel-track');
    const slides = carousel.querySelectorAll('.carousel-slide');
    const dotsContainer = carousel.querySelector('.carousel-dots');

    let currentIndex = 0;

    // CSS values (must match Inside.css) - Desktop
    const SLIDE_WIDTH = 100;
    const GAP = 40;
    const ACTIVE_MARGIN = 200;      // margin: 0 200px for active
    const PREV_NEXT_MARGIN = 60;    // margin: 0 60px for prev/next

    // Tablet values (max-width: 960px)
    const TABLET_SLIDE_WIDTH = 70;
    const TABLET_ACTIVE_MARGIN = 120;
    const TABLET_PREV_NEXT_MARGIN = 30;

    // Mobile values (max-width: 600px)
    const MOBILE_SLIDE_WIDTH = 50;
    const MOBILE_GAP = 25;
    const MOBILE_ACTIVE_MARGIN = 80;
    const MOBILE_PREV_NEXT_MARGIN = 20;

    // Small mobile values (max-width: 400px)
    const SMALL_MOBILE_ACTIVE_MARGIN = 60;
    const SMALL_MOBILE_PREV_NEXT_MARGIN = 15;

    // Create dots
    slides.forEach((_, index) => {
        const dot = document.createElement('span');
        dot.classList.add('carousel-dot');
        if (index === 0) dot.classList.add('active');
        dot.addEventListener('click', () => {
            goToSlide(index);
        });
        dotsContainer.appendChild(dot);
    });

    const dots = dotsContainer.querySelectorAll('.carousel-dot');

    function getPrevIndex() {
        return (currentIndex - 1 + slides.length) % slides.length;
    }

    function getNextIndex() {
        return (currentIndex + 1) % slides.length;
    }

    function getSlideState(index) {
        if (index === currentIndex) return 'active';
        if (index === getPrevIndex()) return 'prev';
        if (index === getNextIndex()) return 'next';
        return 'far';
    }

    function updateSlides() {
        const prevIdx = getPrevIndex();
        const nextIdx = getNextIndex();

        slides.forEach((slide, index) => {
            slide.classList.remove('active', 'prev', 'next', 'far');
            slide.classList.add(getSlideState(index));
        });

        dots.forEach((dot, index) => {
            dot.classList.toggle('active', index === currentIndex);
        });

        centerActiveSlide();
    }

    function centerActiveSlide() {
        const screenWidth = window.innerWidth;

        // Determine breakpoint values
        let activeMargin, prevNextMargin, slideWidth, gap;

        if (screenWidth <= 400) {
            // Small mobile
            activeMargin = SMALL_MOBILE_ACTIVE_MARGIN;
            prevNextMargin = SMALL_MOBILE_PREV_NEXT_MARGIN;
            slideWidth = MOBILE_SLIDE_WIDTH;
            gap = MOBILE_GAP;
        } else if (screenWidth <= 600) {
            // Mobile
            activeMargin = MOBILE_ACTIVE_MARGIN;
            prevNextMargin = MOBILE_PREV_NEXT_MARGIN;
            slideWidth = MOBILE_SLIDE_WIDTH;
            gap = MOBILE_GAP;
        } else if (screenWidth <= 960) {
            // Tablet
            activeMargin = TABLET_ACTIVE_MARGIN;
            prevNextMargin = TABLET_PREV_NEXT_MARGIN;
            slideWidth = TABLET_SLIDE_WIDTH;
            gap = GAP;
        } else {
            // Desktop
            activeMargin = ACTIVE_MARGIN;
            prevNextMargin = PREV_NEXT_MARGIN;
            slideWidth = SLIDE_WIDTH;
            gap = GAP;
        }

        // Calculate the X position of the center of the active slide
        // We iterate through slides in DOM order and sum up their widths + margins + gaps
        let activeCenterX = 0;

        for (let i = 0; i < slides.length; i++) {
            const state = getSlideState(i);

            // Get margins for this slide based on its state
            let leftMargin = 0;
            let rightMargin = 0;

            if (state === 'active') {
                leftMargin = activeMargin;
                rightMargin = activeMargin;
            } else if (state === 'prev' || state === 'next') {
                leftMargin = prevNextMargin;
                rightMargin = prevNextMargin;
            }

            if (i === currentIndex) {
                // Found the active slide - calculate its center
                activeCenterX += leftMargin + (slideWidth / 2);
                break;
            } else {
                // Add this slide's total width (margin + content + margin + gap)
                activeCenterX += leftMargin + slideWidth + rightMargin + gap;
            }
        }

        // Calculate offset to center the active slide
        // Track is positioned at left: 50%, so we just need to shift left by activeCenterX
        const offset = -activeCenterX;

        track.style.transform = `translateX(${offset}px)`;
    }

    function goToSlide(index) {
        currentIndex = index;
        updateSlides();
    }

    function nextSlide() {
        currentIndex = getNextIndex();
        updateSlides();
    }

    // Create fullscreen modal
    const fullscreenModal = document.createElement('div');
    fullscreenModal.className = 'fullscreen-modal';
    fullscreenModal.style.cssText = `
        display: none;
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.95);
        z-index: 9999;
        justify-content: center;
        align-items: center;
        cursor: pointer;
    `;

    const fullscreenImg = document.createElement('img');
    fullscreenImg.style.cssText = `
        max-width: 98%;
        max-height: 98%;
        width: auto;
        height: auto;
        object-fit: contain;
        cursor: default;
    `;

    const closeBtn = document.createElement('button');
    closeBtn.innerHTML = '&times;';
    closeBtn.style.cssText = `
        position: absolute;
        top: 20px;
        right: 30px;
        background: transparent;
        border: none;
        color: #f4e9d5;
        font-size: 3rem;
        cursor: pointer;
        z-index: 10000;
        padding: 0;
        width: auto;
        text-align: center;
        line-height: 1;
    `;

    fullscreenModal.appendChild(fullscreenImg);
    fullscreenModal.appendChild(closeBtn);
    document.body.appendChild(fullscreenModal);

    // Open fullscreen modal
    function openFullscreen(imgSrc) {
        fullscreenImg.src = imgSrc;
        fullscreenModal.style.display = 'flex';
        document.body.style.overflow = 'hidden';
    }

    // Close fullscreen modal
    function closeFullscreen() {
        fullscreenModal.style.display = 'none';
        document.body.style.overflow = '';
    }

    // Close button click
    closeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        closeFullscreen();
    });

    // Click on modal background to close
    fullscreenModal.addEventListener('click', (e) => {
        if (e.target === fullscreenModal) {
            closeFullscreen();
        }
    });

    // Click on slide to focus it or open fullscreen
    slides.forEach((slide, index) => {
        slide.addEventListener('click', (e) => {
            if (index !== currentIndex) {
                // If not active, navigate to it
                e.preventDefault();
                goToSlide(index);
            } else {
                // If active, open fullscreen
                e.preventDefault();
                const img = slide.querySelector('img');
                if (img) {
                    openFullscreen(img.src);
                }
            }
        });
    });

    // Touch/swipe support
    let touchStartX = 0;
    let touchEndX = 0;

    carousel.addEventListener('touchstart', (e) => {
        touchStartX = e.changedTouches[0].screenX;
    }, { passive: true });

    carousel.addEventListener('touchend', (e) => {
        touchEndX = e.changedTouches[0].screenX;
        handleSwipe();
    }, { passive: true });

    function handleSwipe() {
        const swipeThreshold = 50;
        const diff = touchStartX - touchEndX;

        if (Math.abs(diff) > swipeThreshold) {
            if (diff > 0) {
                nextSlide();
            } else {
                currentIndex = getPrevIndex();
                updateSlides();
            }
        }
    }

    // Keyboard navigation and ESC to close fullscreen
    document.addEventListener('keydown', (e) => {
        // ESC to close fullscreen
        if (e.key === 'Escape' && fullscreenModal.style.display === 'flex') {
            closeFullscreen();
            return;
        }

        // Don't navigate if fullscreen is open
        if (fullscreenModal.style.display === 'flex') {
            return;
        }

        // Arrow key navigation
        if (e.key === 'ArrowLeft') {
            currentIndex = getPrevIndex();
            updateSlides();
        } else if (e.key === 'ArrowRight') {
            nextSlide();
        }
    });

    // Initialize
    updateSlides();

    // Handle resize
    window.addEventListener('resize', centerActiveSlide);
});
