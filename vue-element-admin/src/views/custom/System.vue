<template>
  <div :id="id" :class="className" :style="{height:height,width:width}" />
</template>

<script>
import echarts from 'echarts'
import axios from 'axios';
import resize from '@/components/Charts/mixins/resize'

export default {
  mixins: [resize],
  props: {
    className: {
      type: String,
      default: 'system'
    },
    id: {
      type: String,
      default: 'system'
    },
    width: {
      type: String,
      default: '100%'
    },
    height: {
      type: String,
      default: '400px'
    }
  },
  data() {
    return {
      name: "system",
      curTime: new Date(), // 单位是毫秒
      chart: null,
      cpu:[[100000000, 80], [100100000, 82], [100200000, 83], [100300000, 85], [100400000, 82], [100500000, 91], [100600000, 84], [100700000, 90], [100800000, 90], [100900000, 90], [101000000, 95], [101100000, 82], [101200000, 60], [101300000, 82], [101400000, 85], [101500000, 85], [101600000, 92], [101700000, 91], [101800000, 84], [101900000, 90], [102000000, 90], [102100000, 90], [102200000, 85], [102300000, 82]],
      throughtout: [[100000000, 80], [100100000, 82], [100200000, 83], [100300000, 85], [100400000, 82], [100500000, 91], [100600000, 84], [100700000, 90], [100800000, 90], [100900000, 90], [101000000, 95], [101100000, 82], [101200000, 60], [101300000, 82], [101400000, 85], [101500000, 85], [101600000, 92], [101700000, 91], [101800000, 84], [101900000, 90], [102000000, 90], [102100000, 90], [102200000, 85], [102300000, 82]],
    }
  },
  mounted() {
    this.initChart()
  },
  methods: {
    initChart() {
      this.chart = echarts.init(document.getElementById(this.id))
      this.chart.setOption({
        backgroundColor: '#F2F6FC',//'#394056',
        title: {
          top: 20,
          text: '系统资源消耗',
          textStyle: {
            fontWeight: 'normal',
            fontSize: 16,
            color: '#000',//'#F1F1F3'
          },
          left: '1%'
        },
        tooltip: {
          trigger: 'axis',
          axisPointer: {
            lineStyle: {
              color: '#57617B'
            }
          }
        },
        legend: {
          top: 20,
          icon: 'rect',
          itemWidth: 14,
          itemHeight: 5,
          itemGap: 13,
          data: ['带宽', 'CPU'],
          right: '4%',
          textStyle: {
            fontSize: 12,
            color: '#000'//'#F1F1F3'
          },
          selected: {
              '带宽': false,
          }
        },
        // grid: {
        //   top: 100,
        //   left: '2%',
        //   right: '2%',
        //   bottom: '2%',
        //   containLabel: true
        // },
        grid: {
          left: '5%',
          right: '5%',
          borderWidth: 0,
          top: 100,
          bottom: 95,
          textStyle: {
            color: '#fff'
          }
        },
        xAxis: [{
          type: 'time',
          boundaryGap: false,
          axisLine: {
            lineStyle: {
              color: '#57617B'
            }
          },
          axisLabel: {
              interval: 0,
          },
          splitLine: {
            show: false
          },
          axisTick: {
            show: false
          },
          splitArea: {
            show: false
          },
        }],
        yAxis: [{
          type: 'value',
          name: '(%)',
          axisTick: {
            show: false
          },
          axisLine: {
            lineStyle: {
              color: '#57617B'
            }
          },
          axisLabel: {
            margin: 10,
            textStyle: {
              fontSize: 14
            }
          },
          splitLine: {
            lineStyle: {
              color: '#57617B'
            }
          }
        }],
        dataZoom: [{
          show: true,
          height: 30,
          xAxisIndex: [
            0
          ],
          bottom: 30,
          start: 0,
          end: 100,
          handleIcon: 'path://M306.1,413c0,2.2-1.8,4-4,4h-59.8c-2.2,0-4-1.8-4-4V200.8c0-2.2,1.8-4,4-4h59.8c2.2,0,4,1.8,4,4V413z',
          handleSize: '110%',
          handleStyle: {
            color: '#d3dee5'
          },
          textStyle: {
            color: '#fff' },
          borderColor: '#90979c'
        }, {
          type: 'inside',
          show: true,
          height: 15,
          start: 1,
          end: 35
        }],
        series: [{
          name: 'CPU',
          type: 'line',
        //   smooth: true,
          symbol: 'circle',
          symbolSize: 5,
          showSymbol: false,
          lineStyle: {
            normal: {
              width: 1
            }
          },
          areaStyle: {
            normal: {
              color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [{
                offset: 0,
                color: 'rgba(0, 136, 212, 0.3)'
              }, {
                offset: 0.8,
                color: 'rgba(0, 136, 212, 0)'
              }], false),
              shadowColor: 'rgba(0, 0, 0, 0.1)',
              shadowBlur: 10
            }
          },
          itemStyle: {
            normal: {
              color: 'rgb(0,136,212)',
              borderColor: 'rgba(0,136,212,0.2)',
              borderWidth: 12
            }
          },
          data: this.cpu
        }, 
        // {
        //   name: '带宽',
        //   type: 'line',
        // //   smooth: true,
        //   symbol: 'circle',
        //   symbolSize: 5,
        //   showSymbol: false,
        //   lineStyle: {
        //     normal: {
        //       width: 1
        //     }
        //   },
        //   areaStyle: {
        //     normal: {
        //       color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [{
        //         offset: 0,
        //         color: 'rgba(219, 50, 51, 0.3)'
        //       }, {
        //         offset: 0.8,
        //         color: 'rgba(219, 50, 51, 0)'
        //       }], false),
        //       shadowColor: 'rgba(0, 0, 0, 0.1)',
        //       shadowBlur: 10
        //     }
        //   },
        //   itemStyle: {
        //     normal: {
        //       color: 'rgb(219,50,51)',
        //       borderColor: 'rgba(219,50,51,0.2)',
        //       borderWidth: 12
        //     }
        //   },
        //   data: this.throughtout
        // }
        ]
      })
    },
    updateChart() {
      this.chart.setOption({
        series: [{
          data: this.cpu
        }, 
        // {
        //   data: this.throughtout
        // }
        ]
      })
    },
    getData(){
        const path = 'http://127.0.0.1:5000/system';
        axios.get(path)
            .then((res) => {
            let resdata = res.data
            this.throughtout = resdata.bandwidth
            this.cpu = resdata.cpu
            })
            .catch((error) => {
            // eslint-disable-next-line
            console.error(error);
            });
    },
    timer() {
        return setTimeout(() => {
            let now = new Date()
            // console.log(now, now.getSeconds())
            if (now.getSeconds()%1 == 0) {
                this.getData()
                this.updateChart()
                // console.log("come here!")
            }
            this.curTime = now
        }, 5000)
    },
  },
  watch: {
    //   xdata() {
    //       this.timer()
    //   },
    curTime() {
        this.timer()
    }
  },
  created() {
      this.getData()
      this.curTime = new Date()
  },
  destroyed() {
      clearTimeout(this.timer)
  },
}
</script>