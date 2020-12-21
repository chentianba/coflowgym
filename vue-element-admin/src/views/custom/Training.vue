<template>

<div style="height:auto;">
    <iframe style="width:100%;height:100%;" :src="bdTokenUrl" frameborder="0" scrolling="yes" id="bdIframe">
    </iframe>
</div>
</template>

<script>
export default {
    data() {
        return {
            // bdTokenUrl: "http://localhost:6006/",
            // bdTokenUrl: "http://net4:6006/",
            // bdTokenUrl: "http://10.134.147.138:6006/",
            bdTokenUrl: "http://10.134.147.138:6006/#scalars&runSelectionState=eyIyMDIwMTExMlQxNDE4NDQuMTUwMjUxX0REUEdfR0FJTCI6ZmFsc2UsIjIwMjAxMTEyVDIyMjIyOC40NDgxMzlfRERQR19HQUlMIjpmYWxzZSwiMjAyMDExMTNUMTEyNjE4LjYxMTEzOV9ERFBHX0dBSUwiOmZhbHNlfQ%3D%3D"
        }
    },
    mounted(){
        /**
         * iframe-宽高自适应显示   
         */
        const oIframe = document.getElementById('bdIframe');
        const deviceWidth = document.documentElement.clientWidth;
        const deviceHeight = document.documentElement.clientHeight;
        oIframe.style.width = (Number(deviceWidth)-220) + 'px'; //数字是页面布局宽度差值
        oIframe.style.height = (Number(deviceHeight)-120) + 'px'; //数字是页面布局高度差
  },
}
</script>