
let myImage = document.querySelector("img");

myImage.onclick = function () {
  let mySrc = myImage.getAttribute("src");
  if (mySrc === "images/oip.png") {
    myImage.setAttribute("src", "images/firefox2.png");
  } else {
    myImage.setAttribute("src", "images/oip.png");
  }
};
let myButton = document.querySelector("button");
let myHeading = document.querySelector("h1");
function setUserName() {
  let myName = prompt("请输入你的名字。");
  localStorage.setItem("name", myName);
  myHeading.textContent = "Welcome，" + myName;
}
if (!localStorage.getItem("name")) {
  setUserName();
} else {
  let storedName = localStorage.getItem("name");
  myHeading.textContent = "Welcome，" + storedName;
}
myButton.onclick = function () {
  setUserName();
};

function setUserName() {
  let myName = prompt("请输入你的名字。");
  if (!myName) {
    setUserName();
  } else {
    localStorage.setItem("name", myName);
    myHeading.textContent = "Welcome，" + myName;
  }
}

// 函数用于预览上传的图片
function previewImage(event) {
  const image = document.getElementById('imagePreview');
  image.src = URL.createObjectURL(event.target.files[0]);
  image.style.display = 'block';
  predictImage(event.target.files[0]);
}

// 函数用于发送图片到后端进行预测
function predictImage(file) {
  // 可以使用 Fetch API 或 XMLHttpRequest 发送图片到后端服务进行预测
  // 这里只是示例代码，需要后端服务进行处理
  // 这里可以使用 Fetch 或 XMLHttpRequest 发送文件到后端进行处理和预测
  // 在这个函数中，你应该发送文件到你的后端服务，并获取预测结果
  // 假设预测结果是一个 JSON 格式的数据
  const predictionResult = {
    result: '预测结果'
  };

  // 将预测结果显示在页面上
  document.getElementById('predictionResult').innerText = `预测结果：${predictionResult.result}`;
}
