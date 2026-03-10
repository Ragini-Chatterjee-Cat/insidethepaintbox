// Tilted Card Effect - 3D perspective tilt on mouse move
document.addEventListener('DOMContentLoaded', function() {
    const cards = document.querySelectorAll('.tilted-card');

    cards.forEach(card => {
        card.addEventListener('mousemove', handleMove);
        card.addEventListener('mouseleave', handleLeave);
        card.addEventListener('mouseenter', handleEnter);
    });

    function handleMove(e) {
        const card = e.currentTarget;
        const rect = card.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        const centerX = rect.width / 2;
        const centerY = rect.height / 2;

        const rotateX = (y - centerY) / 10;
        const rotateY = (centerX - x) / 10;

        card.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) scale3d(1.05, 1.05, 1.05)`;
    }

    function handleLeave(e) {
        const card = e.currentTarget;
        card.style.transform = 'perspective(1000px) rotateX(0deg) rotateY(0deg) scale3d(1, 1, 1)';
    }

    function handleEnter(e) {
        const card = e.currentTarget;
        card.style.transition = 'transform 0.1s ease-out';
    }
});

// Also apply to .btn elements on homepage
document.addEventListener('DOMContentLoaded', function() {
    const btns = document.querySelectorAll('.btn');

    btns.forEach(btn => {
        btn.style.transition = 'transform 0.3s ease-out';
        btn.style.transformStyle = 'preserve-3d';

        btn.addEventListener('mousemove', function(e) {
            const rect = this.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            const centerX = rect.width / 2;
            const centerY = rect.height / 2;

            const rotateX = (y - centerY) / 20;
            const rotateY = (centerX - x) / 20;

            this.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) scale3d(1.02, 1.02, 1.02)`;
        });

        btn.addEventListener('mouseleave', function() {
            this.style.transform = 'perspective(1000px) rotateX(0deg) rotateY(0deg) scale3d(1, 1, 1)';
        });
    });
});
