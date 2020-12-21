<template>
<div class="app-container">
<!-- <div> -->

<el-form ref="form" :model="form" label-width="120px" :inline="true" class="demo-form-inline">
  <el-form-item label="调度算法">
    <el-select v-model="form.algo" placeholder="请选择调度算法">
      <el-option label="SCF" value="SCF"></el-option>
      <el-option label="SEBF" value="SEBF"></el-option>
      <el-option label="Aalo" value="Aalo"></el-option>
      <el-option label="M-DRL" value="M-DRL"></el-option>
      <el-option label="CS-GAIL" value="CS-GAIL"></el-option>
    </el-select>
  </el-form-item>
  <el-form-item label="Trace数据选择">
  <el-input-number v-model="form.num" @change="handleChange" :min="1" :max="maxTrace" label="描述文字"></el-input-number>
  </el-form-item>
  <el-form-item>
    <el-button type="primary" @click="onSubmit">确定</el-button>
    <!-- <el-button>重置</el-button> -->
  </el-form-item>
</el-form>

<el-table
    ref="table"
    :data="tableData"
    style="width: 100%"
    stripe
    border
    highlight-current-row
    @expand-change="expandChange" :expand-row-keys="expands" :row-key="getRowKeys">
    <el-table-column
        prop="tid"
        label="TraceID"
        align="center"
        width="80">
    </el-table-column>
    <el-table-column
        prop="desc"
        label="描述"
        align="center"
        min-width="150px">
    </el-table-column>
    <el-table-column
        label="Coflow特征"
        align="center"
        width="100"
        type="expand">
      <template slot-scope="scope">
        <el-row class="grid-content" type="flex" align="middle" justify="center">
            <el-col :span="10">
                <mychart :he="subHeight" :cdf="scope.row.cdf"/>
            </el-col>
            <el-col :span="14">
                <h3 align="center">根据Coflow长度和宽度分类</h3>
                <el-table
                    :data="scope.row.dist"
                    style="width: 100%"
                    border
                    fit
                    highlight-current-row>
                    <el-table-column
                        prop="metric"
                        label="指标"
                        align="center"
                        width="140">
                    </el-table-column>
                    <el-table-column
                        prop="sn"
                        label="SN（短窄Coflow）"
                        align="center"
                        min-width="120">
                    </el-table-column>
                    <el-table-column
                        prop="ln"
                        label="LN（长窄Coflow）"
                        align="center"
                        min-width="120">
                    </el-table-column>
                    <el-table-column
                        prop="sw"
                        label="SW（短宽Coflow）"
                        align="center"
                        min-width="120">
                    </el-table-column>
                    <el-table-column
                        prop="lw"
                        label="LW（长宽Coflow）"
                        align="center"
                        min-width="120">
                    </el-table-column>
                </el-table>
            </el-col>
        </el-row>
      </template>
    </el-table-column>
    <el-table-column
        prop="mnum"
        label="机器数量"
        align="center"
        width="180">
    </el-table-column>
    <el-table-column
        prop="cnum"
        label="Coflow数量"
        align="center"
        width="180">
    </el-table-column>
    <el-table-column
        prop="fnum"
        label="流数量"
        align="center"
        width="180">
    </el-table-column>
    <el-table-column
        prop="length"
        label="最长流字节数"
        align="center"
        width="180">
    </el-table-column>
</el-table>

</div>
</template>


<script>
import axios from 'axios';
import mychart from './MyChart'

export default {
    name: 'About',
    components: {mychart},
    data() {
        return {
            expands: [],
            getRowKeys(row) {
                return row.tid
            },
            form: {
                algo: 'M-DRL',
                num: 1,
            },
            maxTrace: 5,
            subHeight: '300px',
            tableData: [{
                tid: 1,
                desc: 'Facebook',
                distname: '重尾分布',
                dist: [{
                    metric: "Coflow数量",
                    sn: '1%',
                    ln: '2%',
                    sw: '3%',
                    lw: '4%',
                }],
                cdf: {
                    data:[[1,1], [2, 0.7], [3, 0.5], [4, 0.3], [5, 0.1]]
                },
                mnum: 150,
                cnum: 526,
                fnum: 1000,
                length: "120KB",
            }, 
            ],
        }
    },
    methods: {
        getTraces() {
            const path = 'http://127.0.0.1:5000/traces';
            console.log(this.tableData);
            axios.get(path)
                .then((res) => {
                this.tableData = res.data.table;
                this.maxTrace = this.tableData.length
                this.form = res.data.form
                })
                .catch((error) => {
                // eslint-disable-next-line
                console.error(error);
                });
        },
        handleChange(value) {
            console.log(value);
        },
        onSubmit() {
            console.log('submit!');
            
            const path = 'http://127.0.0.1:5000/traceSelect'
            axios.post(path, {algo: this.form.algo, traceID: this.form.num})
                .then((res) => {
                    console.log(res)
                    if (res.data.status == "200") {
                        this.$router.push({ path:'/coflow/coflow'})
                        this.$message({
                            message: '成功配置数据源',
                            type: 'success'
                        });
                    } else {
                        this.$message.error('后台有正在运行的Trace');
                    }
                }).catch((error) => {
                    console.error(error);
                });
        },
        expandChange(row, expandedRows) {
            this.expands = []
            if (expandedRows.length > 0) {
                row ? this.expands.push(row.tid) : ''
                this.$refs.table.setCurrentRow(row)
            }
        },
    },
    created() {
        // this.getCoflow();
        this.getTraces();
    }
};
</script>

<style scoped>
.chart-container{
    position: relative;
    width: 100%;
    height: 100%;
    /* height: calc(100vh - 84px); */
}
.el-row {
    margin-bottom: 20px;
    &:last-child {
        margin-bottom: 0;
    }
}
.el-col {
    border-radius: 4px;
}
.bg-purple-dark {
    background: #99a9bf;
}
.bg-purple {
    background: #d3dce6;
}
.bg-purple-light {
    background: #e5e9f2;
}
.grid-content {
    border-radius: 4px;
    min-height: 36px;
}
.row-bg {
    padding: 10px 0;
    background-color: #f9fafc;
    }
</style>