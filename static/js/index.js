window.HELP_IMPROVE_VIDEOJS = false;

$(document).ready(function() {
    // Click boxes
    const boxes = document.querySelectorAll('.system-box');
    const textArea = document.getElementById('system-description-text');

    var currentClickedBox = null;
    function selectBox(box) {
	boxes.forEach(b => b.classList.remove('system-highlighted'));
        if (box != null) {
	  box.classList.add('system-highlighted');
	  textArea.innerText = box.dataset.text;
	}
    };
    boxes.forEach(box => {
	box.addEventListener('mouseover', () => {
	    selectBox(box);
	});
	box.addEventListener('mouseout', () => {
	    selectBox(currentClickedBox);
	});
        box.addEventListener('click', () => {
	    currentClickedBox = box;
	    selectBox(box);
	});
    });
})
