<template>
<div>
    <el-table
        :data="tableData"
        stripe
        height="400"
        highlight-current-row
        :default-sort="{prop: 'sid', order: 'descending'}"
        style="width: width">
        <el-table-column
            prop="sid"
            sortable
            label="Step"
            align="center"
            min-width="80">
        </el-table-column>
        <el-table-column
            prop="acct"
            sortable
            label="平均CCT"
            align="center"
            min-width="100px">
        </el-table-column>
        <el-table-column
            prop="num"
            label="完成Coflow数量"
            align="center"
            min-width="120">
        </el-table-column>
        <el-table-column
            prop="scheduling"
            label="活跃Coflow列表"
            align="center"
            min-width="160">
        </el-table-column>
    </el-table>
</div>
</template>


<script>
import axios from 'axios';
import Chart from './CCTChart'
import mlfq from './MLFQ'
import { List } from 'echarts/lib/export';

export default {
    name: "ccttable",
    props: {
        width: {
            type: String,
            default: '100%'
        },
        height: {
            type: String,
            default: '400px'
        },
        tableData: {
            type: Array,
            default: [{
                sid: 1,
                acct: 100,
                num: 10,
                scheduling: "[1,2,3]",
            },]
        }
    },
    data() {
        return {
        }
    },
    methods: {
    },
    created() {
    },
    destroyed() {
    },
}
</script>
