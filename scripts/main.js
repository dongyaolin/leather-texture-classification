
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
  const formData = new FormData();
  formData.append('file', file);

  // fetch('http://localhost:5000/predict', {
  //   method: 'POST',
  //   body: formData
  // })
  //     .then(response=>response.json())
  //     .then(data => {
  //   // 从后端获取预测结果
  //   const predictionResult = data.result;
  //
  //   // 更新页面上的预测结果元素
  //   const predictionResultDiv = document.getElementById('predictionResult');
  //   predictionResultDiv.textContent = `预测结果为: ${predictionResult}`;
  // })
  //     .catch(error => {
  //   console.error('Error:', error);
  //   // 处理错误情况
  // });


  // .then(response => response.json())
  //
  // .then(data => {
  //     const predictionResultDiv = document.getElementById('predictionResult');
  //     predictionResultDiv.innerHTML = '';
  //     // 从后端获取预测结果数组，假设是一个名为 "similarResults" 的数组
  //     const predictionResult = data.result;
  //
  //     // 创建一个 <ul> 列表，用于展示预测结果
  //     const resultList = document.createElement('ol');
  //
  //     // 将最相似的六个结果添加到列表中
  //     for (let i = 0; i < 6 && i < predictionResult.length; i++) {
  //         const listItem = document.createElement('li');
  //         listItem.textContent = predictionResult[i]; // 假设每个结果是字符串形式
  //         resultList.appendChild(listItem);
  //     }
  //
  //     // 将列表添加到预测结果的 <div> 中
  //     predictionResultDiv.appendChild(resultList);
  // })
  // .catch(error => {
  //     console.error('Error:', error);
  //     // 处理错误情况
  // });
  // }


  fetch('http://localhost:5000/predict', {
    method: 'POST',
    body: formData
  })
  .then(response => response.json())
  .then(data => {
    const predictionResultDiv = document.getElementById('predictionResult');
    predictionResultDiv.innerHTML = ''; // 清空之前的内容

    const orderedList = document.createElement('ol'); // 创建有序列表元素

    data.result.forEach(result => {
      const listItem = document.createElement('li');
      listItem.textContent = `类别: ${result.class}: 概率: ${result.probability}`;
      orderedList.appendChild(listItem); // 将每个结果作为列表项添加到有序列表中
    });

    predictionResultDiv.appendChild(orderedList); // 将有序列表添加到 HTML 元素中
  })
  .catch(error => console.error('Error:', error));
}


