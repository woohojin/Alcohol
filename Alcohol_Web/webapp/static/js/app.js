window.onload = function () {
  const uploaded = document.getElementById("uploaded-image");
  const check = document.getElementById("check");

  check.addEventListener("change", function () {
    if (!check.value) {
      uploaded.classList.remove("visible");
      uploaded.classList.add("invisible");
    } else if (check.value) {
      uploaded.classList.add("visible");
      uploaded.classList.remove("invisible");
    }
  });
};
