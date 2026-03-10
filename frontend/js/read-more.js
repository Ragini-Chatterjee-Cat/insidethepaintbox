// Read More / Read Less functionality for artwork descriptions
document.addEventListener('DOMContentLoaded', function() {
    const readMoreBtns = document.querySelectorAll('.read-more-btn');

    readMoreBtns.forEach(function(btn) {
        btn.addEventListener('click', function() {
            const descriptionText = this.previousElementSibling;

            if (descriptionText.classList.contains('expanded')) {
                descriptionText.classList.remove('expanded');
                this.textContent = 'Read More';
            } else {
                descriptionText.classList.add('expanded');
                this.textContent = 'Read Less';
            }
        });
    });
});
